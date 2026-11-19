//! The lagged fold's decode replay kernels (`new-memory/designs/lagged-fold.md`)
//! on the first device, against an f64 host reference: the conv, the gates
//! and the delta scan of a decode step whose lanes carry 0, 1, 2 and the
//! bound's rows (their inputs in rs pages of a pool, out of page order),
//! keep their state as of none, a carried row, or every row, and land their
//! own row in the rs buffer or not. The window starts at lane 1 of the
//! table. Every output, the carried rows' outputs, the kept states and the
//! rs pages are compared.

use kernels_slang::ssm::{Lag, RsPlane, StatePool};
use kernels_slang::{Segments, Tensor};
use poem_ir::Dtype;

use super::conv1d::{C, W, walk};
use super::gdn_recurrent::{Shape, f64s, gates_of};
use crate::gpu::{Gpu, Out, bf16, landed, noise, widen};

/// The rows a lane carries at most.
const CARRY: usize = 5;
/// Tokens an rs page holds.
const PAGE: u32 = 4;
/// The lag table's lanes; the window is lanes `1..5`.
const TABLE_LANES: usize = 5;
const LANE0: usize = 1;
const SLOTS: usize = 4;
/// Pages a lane's buffer takes, and the token its carried rows start at.
const LANE_PAGES: usize = 3;
const FROM: u32 = 1;

/// One window row: its slot, its carried rows, the rows of its `L + 1`
/// it keeps, and whether its own row lands in the rs buffer.
#[derive(Clone, Copy)]
struct Case {
    slot: usize,
    carried: usize,
    keep: usize,
    own: bool,
}

const CASES: [Case; 4] = [
    Case {
        slot: 2,
        carried: 0,
        keep: 1,
        own: true,
    },
    Case {
        slot: 0,
        carried: 1,
        keep: 0,
        own: true,
    },
    Case {
        slot: 1,
        carried: 2,
        keep: 1,
        own: false,
    },
    Case {
        slot: 3,
        carried: CARRY,
        keep: CARRY + 1,
        own: true,
    },
];

fn t(buf: u32, r: usize, w: usize, d: Dtype) -> Tensor {
    Tensor::new(buf, r as u32, w as u32, d)
}

/// A pool of `state_words` words of state, then rs pages of one plane of
/// `row_words`-word rows: each window lane's pages, out of order.
struct Pages {
    row_words: u32,
    page_words: u32,
    plane_words: u32,
    first_page: u32,
    total_words: usize,
}

impl Pages {
    fn new(state_words: usize, row_words: u32) -> Pages {
        let plane_words = 3;
        let page_words = plane_words + PAGE * row_words + 5;
        let first_page = (state_words as u32).div_ceil(page_words) + 1;
        let total_words =
            ((first_page as usize) + TABLE_LANES * LANE_PAGES + 1) * page_words as usize;
        Pages {
            row_words,
            page_words,
            plane_words,
            first_page,
            total_words,
        }
    }

    /// Window row `r`'s pages, out of order.
    fn pages(&self, r: usize) -> [u32; LANE_PAGES] {
        let base = self.first_page + (LANE_PAGES * r) as u32;
        [base + 2, base, base + 1]
    }

    /// Row `r`'s token `token` as an rs-buffer row.
    fn unit(&self, r: usize, token: u32) -> u32 {
        self.pages(r)[(token / PAGE) as usize] * PAGE + token % PAGE
    }

    /// The first word of rs-buffer row `unit`.
    fn word(&self, unit: u32) -> usize {
        ((unit / PAGE) * self.page_words + self.plane_words + (unit % PAGE) * self.row_words)
            as usize
    }

    fn plane(&self, pool: u32) -> RsPlane {
        RsPlane {
            pool: Segments::one(pool, self.total_words as u32),
            page_tokens: PAGE,
            page_words: self.page_words,
            plane_words: self.plane_words,
            row_words: self.row_words,
        }
    }

    /// The lag table: lane 0 none, then each case's.
    fn table(&self) -> Vec<i32> {
        let ints = Lag::entry_ints(CARRY as u32) as usize;
        let mut table = vec![0i32; TABLE_LANES * ints];
        for (r, case) in CASES.iter().enumerate() {
            let e = &mut table[(LANE0 + r) * ints..][..ints];
            e[0] = case.carried as i32;
            e[1] = case.keep as i32;
            e[2] = if case.own {
                self.unit(r, FROM + case.carried as u32) as i32
            } else {
                -1
            };
            for j in 0..case.carried {
                e[3 + j] = self.unit(r, FROM + j as u32) as i32;
            }
        }
        table
    }
}

/// bf16 rows packed two to a word.
fn packed(values: &[f32]) -> Vec<u32> {
    values
        .chunks_exact(2)
        .map(|p| u32::from(bf16(p[0])) | (u32::from(bf16(p[1])) << 16))
        .collect()
}

/// The f32 each word reads as: what a pool readback compares.
fn as_f32(words: &[u32]) -> Vec<f64> {
    words
        .iter()
        .map(|w| f64::from(f32::from_bits(*w)))
        .collect()
}

fn rounded(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| widen(bf16(*v))).collect()
}

/// Each window row's carried rows' inputs, `width` bf16 elements a row,
/// laid in its pages of `words`; the values as the pool holds them.
fn lay_carried(pages: &Pages, words: &mut [u32], width: usize, seed: u64) -> Vec<Vec<f32>> {
    CASES
        .iter()
        .enumerate()
        .map(|(r, case)| {
            let rows = rounded(&noise(seed + r as u64, case.carried * width));
            for j in 0..case.carried {
                let at = pages.word(pages.unit(r, FROM + j as u32));
                let row = packed(&rows[j * width..][..width]);
                words[at..at + row.len()].copy_from_slice(&row);
            }
            rows
        })
        .collect()
}

/// Window row `r`'s own row lands in its rs-buffer row where it names one.
fn land_own(pages: &Pages, words: &mut [u32], r: usize, row: &[f32]) {
    let case = CASES[r];
    if case.own {
        let at = pages.word(pages.unit(r, FROM + case.carried as u32));
        let row = packed(row);
        words[at..at + row.len()].copy_from_slice(&row);
    }
}

#[test]
fn the_conv1d_replay_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (wb, wr) = gpu.up(&noise(1, C * W));
    let wr = f64s(&wr);
    let wt = t(wb, C, W, Dtype::Bf16);
    let rows = CASES.len();
    for dil in [1usize, 2] {
        let hist = (W - 1) * dil + 1;
        let slab = hist * C;
        let pages = Pages::new(SLOTS * slab, (C / 2) as u32);
        let bank: Vec<f32> = noise(2 + dil as u64, SLOTS * slab);
        let mut words = vec![0u32; pages.total_words];
        for (w, v) in words.iter_mut().zip(&bank) {
            *w = v.to_bits();
        }
        let carried = lay_carried(&pages, &mut words, C, 40 + 10 * dil as u64);
        let (xb, xr) = gpu.up(&noise(30 + dil as u64, rows * C));
        let table = pages.table();

        let mut want_y = vec![0.0; rows * C];
        let mut want_c = vec![0.0; TABLE_LANES * CARRY * C];
        let mut want_words = words.clone();
        let mut want_state = as_f32(&words);
        for (r, case) in CASES.iter().enumerate() {
            let mut seq = f64s(&carried[r]);
            seq.extend(f64s(&xr[r * C..][..C]));
            let s = case.slot;
            let (y, kept) = walk(&f64s(&bank[s * slab..][..slab]), &seq, &wr, dil, case.keep);
            want_y[r * C..][..C].copy_from_slice(&y[case.carried * C..]);
            let lane = LANE0 + r;
            want_c[lane * CARRY * C..][..case.carried * C].copy_from_slice(&y[..case.carried * C]);
            want_state[s * slab..][..slab].copy_from_slice(&kept);
            land_own(&pages, &mut want_words, r, &xr[r * C..][..C]);
        }
        let rs = as_f32(&want_words);
        want_state[SLOTS * slab..].copy_from_slice(&rs[SLOTS * slab..]);

        let tb = gpu.up_i32(&table);
        let sb = gpu.up_u32(&CASES.map(|case| case.slot as u32));
        let words = &words;
        landed(
            &mut gpu,
            &format!("conv1d replay, dilation {dil}"),
            &|g: &mut Gpu| -> Vec<Out> {
                vec![
                    (g.zeros(rows * C), rows * C, false),
                    (
                        g.zeros(TABLE_LANES * CARRY * C),
                        TABLE_LANES * CARRY * C,
                        false,
                    ),
                    (g.up_u32(words), words.len(), true),
                ]
            },
            &[want_y, want_c, want_state],
            &|g, o| {
                let pool = StatePool {
                    bank: t(o[2].0, SLOTS, slab, Dtype::F32),
                    slots: t(sb, rows, 1, Dtype::U32),
                    pool: Segments::one(o[2].0, pages.total_words as u32),
                    origin: 0,
                    slot_stride: slab as u32,
                    pieces: 1,
                    piece_first: 0,
                    piece_atoms: u32::MAX,
                    snapshots: None,
                };
                g.run(|c| {
                    kernels_slang::ssm::conv1d::conv1d_replay(
                        c,
                        t(xb, rows, C, Dtype::Bf16),
                        wt,
                        &pool,
                        &Lag {
                            table: t(tb, TABLE_LANES, table.len() / TABLE_LANES, Dtype::I32),
                            lane0: LANE0 as u32,
                            carry: CARRY as u32,
                        },
                        &pages.plane(o[2].0),
                        W as u32,
                        dil as u32,
                        t(o[0].0, rows, C, Dtype::Bf16),
                        t(o[1].0, TABLE_LANES * CARRY, C, Dtype::Bf16),
                    )
                })
            },
        );
    }
}

#[test]
fn the_gdn_gates_replay_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let heads = 6usize;
    let width = 2 * heads;
    let rows = CASES.len();
    let pages = Pages::new(0, heads as u32);
    let mut words = vec![0u32; pages.total_words];
    let carried = lay_carried(&pages, &mut words, width, 60);
    let (bab, bar) = gpu.up(&noise(61, rows * width));
    let (dtb, dtr) = gpu.up(&noise(62, heads));
    let a_log = noise(63, heads);
    let alb = gpu.up_f32(&a_log);
    let gate = |row: &[f32], out: &mut [f64]| {
        for h in 0..heads {
            let b = f64::from(row[h]);
            let z = f64::from(row[heads + h]) + f64::from(dtr[h]);
            let sp = if z > 20.0 { z } else { z.exp().ln_1p() };
            out[h] = -f64::from(a_log[h]).exp() * sp;
            out[heads + h] = 1.0 / (1.0 + (-b).exp());
        }
    };
    let mut want_g = vec![0.0; rows * width];
    let mut want_c = vec![0.0; TABLE_LANES * CARRY * width];
    let mut want_words = words.clone();
    for (r, case) in CASES.iter().enumerate() {
        gate(
            &bar[r * width..][..width],
            &mut want_g[r * width..][..width],
        );
        for j in 0..case.carried {
            let at = ((LANE0 + r) * CARRY + j) * width;
            gate(
                &carried[r][j * width..][..width],
                &mut want_c[at..at + width],
            );
        }
        land_own(&pages, &mut want_words, r, &bar[r * width..][..width]);
    }
    let table = pages.table();
    let tb = gpu.up_i32(&table);
    let words = &words;
    landed(
        &mut gpu,
        "gdn gates replay",
        &|g: &mut Gpu| -> Vec<Out> {
            vec![
                (g.up_f32(&vec![0.0; rows * width]), rows * width, true),
                (
                    g.up_f32(&vec![0.0; TABLE_LANES * CARRY * width]),
                    TABLE_LANES * CARRY * width,
                    true,
                ),
                (g.up_u32(words), words.len(), true),
            ]
        },
        &[want_g, want_c, as_f32(&want_words)],
        &|g, o| {
            g.run(|c| {
                kernels_slang::ssm::gdn_prep::gdn_prep_replay(
                    c,
                    t(bab, rows, width, Dtype::Bf16),
                    t(dtb, 1, heads, Dtype::Bf16),
                    t(alb, 1, heads, Dtype::F32),
                    t(o[0].0, rows, width, Dtype::F32),
                    &Lag {
                        table: t(tb, TABLE_LANES, table.len() / TABLE_LANES, Dtype::I32),
                        lane0: LANE0 as u32,
                        carry: CARRY as u32,
                    },
                    &pages.plane(o[2].0),
                    t(o[1].0, TABLE_LANES * CARRY, width, Dtype::F32),
                )
            })
        },
    );
}

#[test]
fn the_gdn_replay_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let shapes = [
        Shape {
            k_heads: 1,
            v_heads: 2,
            dk: 128,
            dv: 64,
        },
        Shape {
            k_heads: 2,
            v_heads: 4,
            dk: 32,
            dv: 48,
        },
    ];
    let rows = CASES.len();
    let held = TABLE_LANES * CARRY;
    let pages = Pages::new(0, 1);
    let table = pages.table();
    for (n, sh) in shapes.into_iter().enumerate() {
        for &tier in crate::gpu::tiers() {
            let tactic = kernels_slang::ssm::gdn_recurrent::Tactic {
                tier,
                register_min_rows: 0,
            };
            let slab = sh.slab();
            let bank: Vec<f32> = noise(70 + n as u64, SLOTS * slab)
                .iter()
                .map(|v| v * 0.25)
                .collect();
            let (qb, qr) = gpu.up(&noise(71 + n as u64, rows * sh.pitch()));
            let gr = gates_of(72 + n as u64, rows, sh.v_heads);
            let gb = gpu.up_f32(&gr);
            let (cqb, cqr) = gpu.up(&noise(73 + n as u64, held * sh.pitch()));
            let cgr = gates_of(74 + n as u64, held, sh.v_heads);
            let cgb = gpu.up_f32(&cgr);
            let (qr, gr, cqr, cgr) = (f64s(&qr), f64s(&gr), f64s(&cqr), f64s(&cgr));

            let y_n = rows * sh.v_heads * sh.dv;
            let mut want_y = vec![0.0; y_n];
            let mut unread = vec![0.0; held * sh.v_heads * sh.dv];
            let mut want_s = f64s(&bank);
            for (r, case) in CASES.iter().enumerate() {
                let mut s = f64s(&bank[case.slot * slab..][..slab]);
                let lane = LANE0 + r;
                for j in 0..=case.carried {
                    if j < case.carried {
                        sh.token(&cqr, &cgr, lane * CARRY + j, &mut s, &mut unread);
                    } else {
                        let mut own = vec![0.0; y_n];
                        sh.token(&qr, &gr, r, &mut s, &mut own);
                        let at = r * sh.v_heads * sh.dv;
                        want_y[at..at + sh.v_heads * sh.dv]
                            .copy_from_slice(&own[at..at + sh.v_heads * sh.dv]);
                    }
                    if j + 1 == case.keep {
                        want_s[case.slot * slab..][..slab].copy_from_slice(&s);
                    }
                }
            }
            let tb = gpu.up_i32(&table);
            let sb = gpu.up_u32(&CASES.map(|case| case.slot as u32));
            let bank = &bank;
            let work_n = TABLE_LANES * slab;
            let (kh, vh, dk, dv) = (
                sh.k_heads as u32,
                sh.v_heads as u32,
                sh.dk as u32,
                sh.dv as u32,
            );
            landed(
                &mut gpu,
                &format!("gdn replay, k {} at {tier:?}", sh.dk),
                &|g: &mut Gpu| -> Vec<Out> {
                    vec![
                        (g.up_f32(&vec![0.0; y_n]), y_n, true),
                        (g.up_f32(bank), SLOTS * slab, true),
                        (g.up_f32(&vec![0.0; work_n]), 0, true),
                    ]
                },
                &[want_y, want_s],
                &|g, o| {
                    let pool = StatePool::bank(
                        t(o[1].0, SLOTS, slab, Dtype::F32),
                        t(sb, rows, 1, Dtype::U32),
                    );
                    g.run(|c| {
                        kernels_slang::ssm::gdn_recurrent::gated_delta_replay(
                            c,
                            tactic,
                            t(qb, rows, sh.pitch(), Dtype::Bf16),
                            t(gb, rows, 2 * sh.v_heads, Dtype::F32),
                            &pool,
                            &Lag {
                                table: t(tb, TABLE_LANES, table.len() / TABLE_LANES, Dtype::I32),
                                lane0: LANE0 as u32,
                                carry: CARRY as u32,
                            },
                            t(cqb, held, sh.pitch(), Dtype::Bf16),
                            t(cgb, held, 2 * sh.v_heads, Dtype::F32),
                            Some(t(o[2].0, TABLE_LANES, slab, Dtype::F32)),
                            kh,
                            vh,
                            dk,
                            dv,
                            t(o[0].0, rows, sh.v_heads * sh.dv, Dtype::F32),
                        )
                    })
                },
            );
        }
    }
}

/// The decode step's recurrent kernels of every delta layer of
/// Qwen3.5-0.8B (18 layers: a 6144-channel conv of width 4, 16 value heads
/// of 128 x 128), each layer's state and rs pages its own, timed as one
/// frame: the plain decode kernels, and the replay kernels with every lane
/// carrying 0, 1 and 2 rows (the bound the engine lays at two frames in
/// flight, 20, sizing the carried scratch), at 1 and 64 lanes. A
/// measurement, not a check: `-- --ignored --nocapture
/// the_replay_kernels_cost_at_serving_shapes`.
#[test]
#[ignore = "a measurement of the replay kernels at Qwen3.5-0.8B's shapes"]
fn the_replay_kernels_cost_at_serving_shapes() {
    const LAYERS: usize = 18;
    const CH: usize = 6144;
    const TAPS: usize = 4;
    const VH: usize = 16;
    const KH: usize = 16;
    const DK: usize = 128;
    const DV: usize = 128;
    const BOUND: usize = 20;
    const PASSES: usize = 8;
    const ROUNDS: usize = 7;
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let tactic = kernels_slang::ssm::gdn_recurrent::Tactic {
        tier: *crate::gpu::tiers().last().expect("a tier"),
        register_min_rows: 0,
    };
    let conv_slab = TAPS * CH;
    let gdn_slab = VH * DK * DV;
    let (conv_row, ba_row) = ((CH / 2) as u32, VH as u32);
    let page_words = PAGE * (conv_row + ba_row);
    for lanes in [1usize, 64] {
        let state_words = lanes * (conv_slab + gdn_slab);
        let first_page = (state_words as u32).div_ceil(page_words);
        let total = (first_page as usize + 2 * lanes) * page_words as usize;
        let pools: Vec<u32> = (0..LAYERS).map(|_| gpu.zeros(2 * total)).collect();
        let (xb, _) = gpu.up(&noise(1, lanes * CH));
        let (wb, _) = gpu.up(&noise(2, CH * TAPS));
        let (bab, _) = gpu.up(&noise(3, lanes * 2 * VH));
        let (dtb, _) = gpu.up(&noise(4, VH));
        let alb = gpu.up_f32(&noise(5, VH));
        let qkv = gpu.zeros(lanes * CH);
        let gates = gpu.up_f32(&vec![0.0; lanes * 2 * VH]);
        let y = gpu.up_f32(&vec![0.0; lanes * VH * DV]);
        let cq = gpu.zeros(lanes * BOUND * CH);
        let cg = gpu.up_f32(&vec![0.0; lanes * BOUND * 2 * VH]);
        let slots = gpu.up_u32(&(0..lanes as u32).collect::<Vec<_>>());
        let ints = Lag::entry_ints(BOUND as u32) as usize;
        let tables: Vec<u32> = (0..3usize)
            .map(|carried| {
                let mut table = vec![0i32; lanes * ints];
                for lane in 0..lanes {
                    let unit = |token: u32| {
                        ((first_page + 2 * lane as u32 + token / PAGE) * PAGE + token % PAGE) as i32
                    };
                    let e = &mut table[lane * ints..][..ints];
                    e[0] = carried as i32;
                    e[1] = 1;
                    e[2] = unit(FROM + carried as u32);
                    for j in 0..carried {
                        e[3 + j] = unit(FROM + j as u32);
                    }
                }
                gpu.up_i32(&table)
            })
            .collect();
        let pool_of = |pool: u32, origin: usize, slab: usize| StatePool {
            bank: t(pool, lanes, slab, Dtype::F32),
            slots: t(slots, lanes, 1, Dtype::U32),
            pool: Segments::one(pool, total as u32),
            origin: origin as u32,
            slot_stride: slab as u32,
            pieces: 1,
            piece_first: 0,
            piece_atoms: u32::MAX,
            snapshots: None,
        };
        let plane = |pool: u32, at: u32, row: u32| RsPlane {
            pool: Segments::one(pool, total as u32),
            page_tokens: PAGE,
            page_words,
            plane_words: at,
            row_words: row,
        };
        // The mode: none (the plain decode kernels), or the carried rows.
        let pass = |g: &Gpu, mode: Option<usize>| {
            g.run(|c| {
                for _ in 0..PASSES {
                    for &pool in &pools {
                        let conv = pool_of(pool, 0, conv_slab);
                        let gdn = pool_of(pool, lanes * conv_slab, gdn_slab);
                        let (x, w) = (t(xb, lanes, CH, Dtype::Bf16), t(wb, CH, TAPS, Dtype::Bf16));
                        let q = t(qkv, lanes, CH, Dtype::Bf16);
                        let ba = t(bab, lanes, 2 * VH, Dtype::Bf16);
                        let (dt, al) = (t(dtb, 1, VH, Dtype::Bf16), t(alb, 1, VH, Dtype::F32));
                        let gt = t(gates, lanes, 2 * VH, Dtype::F32);
                        let yt = t(y, lanes, VH * DV, Dtype::F32);
                        match mode {
                            None => {
                                kernels_slang::ssm::conv1d::conv1d(
                                    c,
                                    x,
                                    w,
                                    &conv,
                                    TAPS as u32,
                                    1,
                                    q,
                                )?;
                                kernels_slang::ssm::gdn_prep::gdn_prep(c, ba, dt, al, gt)?;
                                kernels_slang::ssm::gdn_recurrent::gated_delta(
                                    c, tactic, q, gt, &gdn, KH as u32, VH as u32, DK as u32,
                                    DV as u32, yt,
                                )?;
                            }
                            Some(carried) => {
                                let lag = Lag {
                                    table: t(tables[carried], lanes, ints, Dtype::I32),
                                    lane0: 0,
                                    carry: BOUND as u32,
                                };
                                let cqt = t(cq, lanes * BOUND, CH, Dtype::Bf16);
                                let cgt = t(cg, lanes * BOUND, 2 * VH, Dtype::F32);
                                kernels_slang::ssm::conv1d::conv1d_replay(
                                    c,
                                    x,
                                    w,
                                    &conv,
                                    &lag,
                                    &plane(pool, 0, conv_row),
                                    TAPS as u32,
                                    1,
                                    q,
                                    cqt,
                                )?;
                                kernels_slang::ssm::gdn_prep::gdn_prep_replay(
                                    c,
                                    ba,
                                    dt,
                                    al,
                                    gt,
                                    &lag,
                                    &plane(pool, PAGE * conv_row, ba_row),
                                    cgt,
                                )?;
                                kernels_slang::ssm::gdn_recurrent::gated_delta_replay(
                                    c, tactic, q, gt, &gdn, &lag, cqt, cgt, None, KH as u32,
                                    VH as u32, DK as u32, DV as u32, yt,
                                )?;
                            }
                        }
                    }
                }
                Ok(())
            });
        };
        let modes = [None, Some(0), Some(1), Some(2)];
        for &mode in &modes {
            pass(&gpu, mode);
        }
        // Each mode's wall time a pass (encoding, submit and wait), and its
        // kernels' device time a pass.
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
        let mut device: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
        let mut parts: Vec<Vec<[f64; 3]>> = vec![Vec::new(); modes.len()];
        for _ in 0..ROUNDS {
            for (m, &mode) in modes.iter().enumerate() {
                crate::backend::time_kernels(true);
                let start = std::time::Instant::now();
                pass(&gpu, mode);
                times[m].push(start.elapsed().as_secs_f64() * 1e6 / PASSES as f64);
                let rows = crate::backend::kernel_times();
                let of = |part: &str| -> f64 {
                    rows.iter()
                        .filter(|(name, _, _)| name.contains(part))
                        .map(|row| row.1 as f64)
                        .sum::<f64>()
                        / 1e3
                        / PASSES as f64
                };
                crate::backend::time_kernels(false);
                device[m].push(of("ssm"));
                parts[m].push([of("conv1d"), of("gdn_prep"), of("gdn_recurrent")]);
            }
        }
        let median = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (base, base_dev) = (median(&times[0]), median(&device[0]));
        for (m, mode) in modes.iter().enumerate() {
            let (us, dev) = (median(&times[m]), median(&device[m]));
            let part = |k: usize| median(&parts[m].iter().map(|p| p[k]).collect::<Vec<_>>());
            let name = match mode {
                None => "plain decode".to_string(),
                Some(l) => format!("replay L={l}"),
            };
            eprintln!(
                "LAGBENCH {} c{lanes}: {name}: device {dev:.1} us ({:+.1}% of plain; conv {:.1}, \
                 gates {:.1}, scan {:.1}), wall {us:.1} us ({:+.1}%), a step's 18 delta layers",
                crate::backend::NAME,
                (dev / base_dev - 1.0) * 100.0,
                part(0),
                part(1),
                part(2),
                (us / base - 1.0) * 100.0
            );
        }
    }
}
