//! Paged attention on the first device against an f64 host reference:
//! decode a row a workgroup on the subgroup tier and the baseline, 256 and
//! 128 invocations, plain, windowed, masked (from a committed boundary on,
//! the keys below it read whatever the plane says), with its LSE, and split
//! into key ranges folded by their LSEs (rows a single split covers among
//! them); the selected form over named key blocks; prefill tiled at 32, 16
//! and 8 rows (16 lanes a row too, folded by shuffles where subgroups hold
//! 16 lanes, 32 on WGSL), a request straddling a tile, and walked a row a
//! workgroup (the few-rows arm), with and without its LSE, its second
//! request masked past a boundary.

use kernels_slang::Tensor;
use kernels_slang::attn::sdpa;
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

const PAGE: usize = 16;
const QH: usize = 8;
const KVH: usize = 2;

struct Batch {
    positions: Vec<i32>,
    owner: Vec<i32>,
    /// Row `r` reads its mask when `masks[r]` is some: its lane's committed
    /// boundary, every key below it read under the causal bound and the
    /// window alone, and `MASK_STRIDE` bytes, column `c` key `origin + c`.
    masks: Vec<Option<(usize, Vec<u8>)>>,
}

/// Request 0 reads pages [3, 0, 5], request 1 pages [1, 2].
const PAGES: [u32; 5] = [3, 0, 5, 1, 2];
const INDPTR: [u32; 3] = [0, 3, 5];
const MASK_STRIDE: usize = 48;

/// The selected form's block width and its blocks a row.
const RATIO: usize = 8;
const TOP_K: usize = 3;

fn slot(req: usize, pos: usize) -> usize {
    PAGES[INDPTR[req] as usize + pos / PAGE] as usize * PAGE + pos % PAGE
}

/// Whether row `r` reads key `kp`: causal, inside the window and, past its
/// boundary, the mask.
fn keeps(b: &Batch, r: usize, kp: usize, window: Option<usize>) -> bool {
    let pos = b.positions[r] as usize;
    let start = window.map_or(0, |w| (pos + 1).saturating_sub(w));
    (start..=pos).contains(&kp)
        && b.masks[r].as_ref().is_none_or(|(origin, m)| {
            kp < *origin || (kp - origin < MASK_STRIDE && m[kp - origin] != 0)
        })
}

/// A sparse plane past a boundary: hides some buffered keys of every row and
/// keeps the row's own; set where it reaches past the row, which the causal
/// bound hides.
fn sparse(origin: usize, pos: usize, row: usize) -> Vec<u8> {
    (0..MASK_STRIDE)
        .map(|c| u8::from(origin + c >= pos || !(c * 7 + row * 3).is_multiple_of(5)))
        .collect()
}

fn causal(b: &Batch, window: Option<usize>) -> Vec<Vec<usize>> {
    (0..b.positions.len())
        .map(|r| {
            (0..=b.positions[r] as usize)
                .filter(|&kp| keeps(b, r, kp, window))
                .collect()
        })
        .collect()
}

/// The keys of the blocks `sel` names (a negative id or one past the row's
/// whole blocks skipped), then the row's open block.
fn selected(b: &Batch, sel: &[i32]) -> Vec<Vec<usize>> {
    (0..b.positions.len())
        .map(|r| {
            let pos = b.positions[r] as usize;
            let blocks = (pos + 1) / RATIO;
            let mut keys = Vec::new();
            for &c in &sel[r * TOP_K..][..TOP_K.min(blocks)] {
                if c >= 0 && (c as usize) < blocks {
                    keys.extend(c as usize * RATIO..(c as usize + 1) * RATIO);
                }
            }
            keys.extend(blocks * RATIO..blocks * RATIO + RATIO - 1);
            keys.retain(|&kp| keeps(b, r, kp, None));
            keys
        })
        .collect()
}

/// Each row's attention over `keys[r]`, and its base-2 LSE.
fn reference(
    b: &Batch,
    keys: &[Vec<usize>],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    hd: usize,
    scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let rows = b.positions.len();
    let mut o = vec![0.0; rows * QH * hd];
    let mut l = vec![0.0; rows * QH];
    for r in 0..rows {
        for h in 0..QH {
            let kh = h / (QH / KVH);
            let qv = &q[(r * QH + h) * hd..][..hd];
            let at = |kp: usize| (slot(b.owner[r] as usize, kp) * KVH + kh) * hd;
            let s: Vec<f64> = keys[r]
                .iter()
                .map(|&kp| {
                    qv.iter()
                        .zip(&k[at(kp)..][..hd])
                        .map(|(a, c)| a * c)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - top).exp()).sum();
            l[r * QH + h] = (z.ln() + top) * std::f64::consts::LOG2_E;
            for d in 0..hd {
                o[(r * QH + h) * hd + d] = keys[r]
                    .iter()
                    .zip(&s)
                    .map(|(&kp, x)| (x - top).exp() * v[at(kp) + d])
                    .sum::<f64>()
                    / z;
            }
        }
    }
    (o, l)
}

/// What the Slang kernels are asked, beside the arm it stands for.
#[derive(Clone, Copy, Debug)]
enum Form {
    Decode,
    /// Decode split at most `SPLITS` ways, `MIN_KEYS` keys a split.
    Split,
    Selected,
    Tiled,
}

const SPLITS: usize = 8;
const MIN_KEYS: u32 = 8;

#[test]
fn paged_attention_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let slots = 6 * PAGE;
    let (pi, pp) = (gpu.up_u32(&PAGES), gpu.up_u32(&INDPTR));
    let decode = Batch {
        positions: vec![39, 20, 5, 30],
        owner: vec![0, 1, 0, 0],
        masks: vec![
            None,
            // A boundary past half the row's keys, the window's start
            // among the committed ones and among the masked ones.
            Some((12, sparse(12, 20, 1))),
            None,
            Some((0, (0..MASK_STRIDE).map(|i| u8::from(i % 3 != 1)).collect())),
        ],
    };
    // Row 0 names blocks 3, 0 and a block past its whole ones; row 1 one
    // block of its two and a negative id; row 2 has no whole block.
    let selection: Vec<i32> = vec![3, 0, 4, 1, -1, 0, 0, 1, 2, 2, 0, 1];
    let prefill = Batch {
        positions: (0..26).chain(0..11).collect(),
        owner: std::iter::repeat_n(0, 26)
            .chain(std::iter::repeat_n(1, 11))
            .collect(),
        // Request 1's rows read through a mask past a boundary at 4: its
        // committed keys whatever the plane says, then the plane.
        masks: std::iter::repeat_n(None, 26)
            .chain((0..11).map(|pos| Some((4, sparse(4, pos, pos)))))
            .collect(),
    };
    let narrow = gpu.subgroup() < 32;
    for hd in [64usize, 128] {
        let scale = 1.0 / (hd as f64).sqrt();
        let one = gpu.one_segment();
        let ((keys, kr), (values, vr)) = gpu.up_planes(
            &noise(hd as u64, slots * KVH * hd),
            &noise(hd as u64 + 1, slots * KVH * hd),
        );
        let (kr, vr) = (f64s(&kr), f64s(&vr));
        let pool = crate::gpu::kv_planes(
            one,
            t(keys, slots, KVH * hd, Dtype::Bf16),
            t(values, slots, KVH * hd, Dtype::Bf16),
            t(pi, PAGES.len(), 1, Dtype::U32),
            t(pp, 3, 1, Dtype::U32),
            PAGE as i32,
            (KVH * hd) as u64,
            hd as u64,
        );
        // Packed (one segment), the pool is its own copy: its values plane
        // names the keys' buffer at a word past them.
        let spool = if one {
            pool
        } else {
            crate::gpu::kv_planes(
                one,
                pool.keys,
                pool.values,
                pool.page_indices,
                pool.page_indptr,
                pool.page_size,
                pool.seq_stride,
                pool.head_stride,
            )
        };
        for (what, batch, prefilling) in [("decode", &decode, false), ("prefill", &prefill, true)] {
            let rows = batch.positions.len();
            let (q, qr) = gpu.up(&noise(hd as u64 + rows as u64, rows * QH * hd));
            let qr = f64s(&qr);
            let pos = gpu.up_i32(&batch.positions);
            let own = gpu.up_i32(&batch.owner);
            let sel = gpu.up_i32(&selection);
            let mut mask = vec![0u8; rows * MASK_STRIDE];
            // Two words a row: its boundary (none past every key) and where
            // its columns start in the plane.
            let mut words = [sdpa::MASK_NONE, 0].repeat(rows);
            for (r, m) in batch.masks.iter().enumerate() {
                if let Some((origin, m)) = m {
                    mask[r * MASK_STRIDE..][..MASK_STRIDE].copy_from_slice(m);
                    words[2 * r] = *origin as u32;
                    words[2 * r + 1] = (r * MASK_STRIDE) as u32;
                }
            }
            let pack = |b: &[u8]| -> Vec<u32> {
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
            };
            let (mb, eb) = (gpu.up_u32(&pack(&mask)), gpu.up_u32(&words));
            let tables = sdpa::Tables {
                positions: t(pos, rows, 1, Dtype::I32),
                request_of_token: t(own, rows, 1, Dtype::I32),
                mask: t(mb, 1, rows * MASK_STRIDE, Dtype::U8),
                mask_words: t(eb, rows, 2, Dtype::U32),
                row0: 0,
                table0: 0,
                rows: rows as u32,
            };
            let n = rows * QH * hd;
            let qt = t(q, rows, QH * hd, Dtype::Bf16);
            let (part_out, part_lse) = (
                gpu.up_f32(&vec![0.0; SPLITS * n]),
                gpu.up_f32(&vec![0.0; SPLITS * rows * QH]),
            );
            let split = sdpa::Split {
                part_out: t(part_out, SPLITS * rows * QH, hd, Dtype::F32),
                part_lse: t(part_lse, SPLITS * rows, QH, Dtype::F32),
            };

            let forms: &[Form] = if prefilling {
                &[Form::Tiled, Form::Decode]
            } else {
                &[Form::Decode, Form::Split, Form::Selected]
            };
            let windows: &[Option<usize>] = &[None, Some(16)];
            for &form in forms {
                for &window in windows {
                    if matches!(form, Form::Selected) && window.is_some() {
                        continue;
                    }
                    let keys = match form {
                        Form::Selected => selected(batch, &selection),
                        _ => causal(batch, window),
                    };
                    let (want_o, want_l) = reference(batch, &keys, &qr, &kr, &vr, hd, scale);
                    let win = window.map(|w| w as u32);
                    let shape = format!("{what} {form:?} d {hd} window {window:?}");

                    let tactics: &[(bool, u32, u32, u32)] = if matches!(form, Form::Tiled) {
                        &[
                            (false, 256, 32, 32),
                            (false, 256, 16, 32),
                            (false, 256, 8, 32),
                            (false, 256, 16, 16),
                            (true, 256, 16, 16),
                        ]
                    } else {
                        &[
                            (true, 256, 32, 32),
                            (true, 128, 32, 32),
                            (false, 256, 32, 32),
                            (false, 128, 32, 32),
                        ]
                    };
                    for &(subgroup, threads, tile, lanes) in tactics {
                        // The subgroup tile folds a row's 16 lanes in one
                        // subgroup; WGSL's subgroup kernels fold 32.
                        let floor = if !crate::backend::SPIRV {
                            32
                        } else if lanes == 16 {
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
                            tile_lanes: lanes,
                            split_groups: 1024,
                            split_min_keys: MIN_KEYS,
                            prefill_work: 0,
                        };
                        let lses: &[bool] = if matches!(form, Form::Selected) {
                            &[false]
                        } else {
                            &[false, true]
                        };
                        let works = lses.iter().flat_map(|&lse| [(lse, 0), (lse, 1)]);
                        for (with_lse, prefill_work) in works {
                            let tactic = sdpa::Tactic {
                                prefill_work,
                                ..tactic
                            };
                            let fire =
                                |c: &kernels_slang::Ctx<'_>,
                                 tables: &sdpa::Tables,
                                 ot: Tensor,
                                 lse: Option<Tensor>| match form {
                                    Form::Tiled => sdpa::prefill(
                                        c,
                                        tactic,
                                        qt,
                                        tables,
                                        &spool,
                                        win,
                                        hd as u32,
                                        scale as f32,
                                        ot,
                                        lse,
                                    ),
                                    Form::Selected => sdpa::selected(
                                        c,
                                        tactic,
                                        qt,
                                        tables,
                                        &spool,
                                        hd as u32,
                                        scale as f32,
                                        RATIO as u32,
                                        t(sel, rows, TOP_K, Dtype::I32),
                                        ot,
                                    ),
                                    Form::Decode | Form::Split => sdpa::decode(
                                        c,
                                        tactic,
                                        qt,
                                        tables,
                                        &spool,
                                        win,
                                        hd as u32,
                                        scale as f32,
                                        ot,
                                        lse,
                                        matches!(form, Form::Split).then_some(split),
                                    ),
                                };
                            let label = format!(
                                "{shape} subgroup {subgroup} threads {threads} rows {tile} lanes {lanes} lse {with_lse}"
                            );
                            // The fire whole, then cut in two windows over the
                            // same whole planes, the second from row `cut` on:
                            // a window's rows land what the whole fire's do.
                            let cut = (rows / 3) as u32;
                            let windows: [&[(u32, u32)]; 2] =
                                [&[(0, rows as u32)], &[(0, cut), (cut, rows as u32 - cut)]];
                            for (cuts, spans) in windows.iter().enumerate() {
                                let (o, l) = (gpu.zeros(n), gpu.up_f32(&vec![0.0; rows * QH]));
                                let lse = with_lse.then(|| t(l, rows, QH, Dtype::F32));
                                let ot = t(o, rows, QH * hd, Dtype::Bf16);
                                for &(row0, span) in spans.iter() {
                                    let window = sdpa::Tables {
                                        row0,
                                        table0: row0,
                                        rows: span,
                                        ..tables
                                    };
                                    gpu.run(|c| fire(c, &window, ot, lse));
                                }
                                let label = format!("{label} windows {}", cuts + 1);
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
        }
    }
}
