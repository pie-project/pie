//! Paged attention on the first wgpu adapter against an f64 host reference:
//! decode a row a workgroup on the baseline tier, 256 and 128 invocations, plain, windowed, masked,
//! with its LSE, and split into key ranges folded by their LSEs (rows a
//! single split covers among them); the selected form over named key
//! blocks; prefill tiled at 32, 16 and 8 rows (16 lanes a row too, folded
//! by shuffles where every subgroup is 32 lanes), a request straddling a tile,
//! and walked a row a workgroup (the few-rows arm), with and without its
//! LSE.

use kernels_slang::attn::sdpa;
use kernels_slang::{KvPool, Tensor};
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
    /// Row `r` reads its mask when `masks[r]` is some, `mask_stride` bytes.
    masks: Vec<Option<Vec<u8>>>,
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

/// Whether row `r` reads key `kp`: causal, inside the window and the mask.
fn keeps(b: &Batch, r: usize, kp: usize, window: Option<usize>) -> bool {
    let pos = b.positions[r] as usize;
    let start = window.map_or(0, |w| (pos + 1).saturating_sub(w));
    (start..=pos).contains(&kp)
        && b.masks[r]
            .as_ref()
            .is_none_or(|m| kp < MASK_STRIDE && m[kp] != 0)
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
    let subgroups = gpu.device.chip().subgroups_32();
    let slots = 6 * PAGE;
    let (pi, pp) = (gpu.up_u32(&PAGES), gpu.up_u32(&INDPTR));
    let decode = Batch {
        positions: vec![39, 20, 5, 30],
        owner: vec![0, 1, 0, 0],
        masks: vec![
            None,
            None,
            None,
            Some((0..MASK_STRIDE).map(|i| u8::from(i % 3 != 1)).collect()),
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
        masks: vec![None; 37],
    };
    for hd in [64usize, 128] {
        let scale = 1.0 / (hd as f64).sqrt();
        let (keys, kr) = gpu.up(&noise(hd as u64, slots * KVH * hd));
        let (values, vr) = gpu.up(&noise(hd as u64 + 1, slots * KVH * hd));
        let (kr, vr) = (f64s(&kr), f64s(&vr));
        let pool = KvPool {
            keys: t(keys, slots, KVH * hd, Dtype::Bf16),
            values: t(values, slots, KVH * hd, Dtype::Bf16),
            page_indices: t(pi, PAGES.len(), 1, Dtype::U32),
            page_indptr: t(pp, 3, 1, Dtype::U32),
            page_size: PAGE as i32,
            seq_stride: (KVH * hd) as u64,
            head_stride: hd as u64,
        };
        let spool = kernels_slang::KvPool {
            keys: pool.keys,
            values: pool.values,
            page_indices: pool.page_indices,
            page_indptr: pool.page_indptr,
            page_size: pool.page_size,
            seq_stride: pool.seq_stride,
            head_stride: pool.head_stride,
        };
        for (what, batch, prefilling) in [("decode", &decode, false), ("prefill", &prefill, true)] {
            let rows = batch.positions.len();
            let (q, qr) = gpu.up(&noise(hd as u64 + rows as u64, rows * QH * hd));
            let qr = f64s(&qr);
            let pos = gpu.up_i32(&batch.positions);
            let own = gpu.up_i32(&batch.owner);
            let sel = gpu.up_i32(&selection);
            let mut mask = vec![0u8; rows * MASK_STRIDE];
            let mut enabled = vec![0u8; rows];
            for (r, m) in batch.masks.iter().enumerate() {
                if let Some(m) = m {
                    mask[r * MASK_STRIDE..][..MASK_STRIDE].copy_from_slice(m);
                    enabled[r] = 1;
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
            let (mb, eb) = (gpu.up_u32(&pack(&mask)), gpu.up_u32(&pack(&enabled)));
            let tables = sdpa::Tables {
                positions: t(pos, rows, 1, Dtype::I32),
                request_of_token: t(own, rows, 1, Dtype::I32),
                mask: t(mb, rows, MASK_STRIDE, Dtype::U8),
                mask_enabled: t(eb, rows, 1, Dtype::U8),
                mask_stride: MASK_STRIDE as u32,
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

                    // At the subgroup tier the 16-lane tile folds by
                    // shuffles; decode stays the baseline's on WGSL.
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
                            (false, 256, 32, 32),
                            (false, 128, 32, 32),
                            (true, 256, 32, 32),
                        ]
                    };
                    for &(subgroup, threads, tile, lanes) in tactics {
                        if subgroup && !subgroups {
                            continue;
                        }
                        let tactic = sdpa::Tactic {
                            tier: if subgroup {
                                kernels_slang::Tier::Subgroup
                            } else {
                                kernels_slang::Tier::Baseline
                            },
                            narrow: false,
                            decode_threads: threads,
                            tile_rows: tile,
                            tile_lanes: lanes,
                            split_groups: 1024,
                            split_min_keys: MIN_KEYS,
                        };
                        let lses: &[bool] = if matches!(form, Form::Selected) {
                            &[false]
                        } else {
                            &[false, true]
                        };
                        for &with_lse in lses {
                            let (o, l) = (gpu.zeros(n), gpu.up_f32(&vec![0.0; rows * QH]));
                            let lse = with_lse.then(|| t(l, rows, QH, Dtype::F32));
                            let ot = t(o, rows, QH * hd, Dtype::Bf16);
                            gpu.run(|c| match form {
                                Form::Tiled => sdpa::prefill(
                                    c,
                                    tactic,
                                    qt,
                                    &tables,
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
                                    &tables,
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
                                    &tables,
                                    &spool,
                                    win,
                                    hd as u32,
                                    scale as f32,
                                    ot,
                                    lse,
                                    matches!(form, Form::Split).then_some(split),
                                ),
                            });
                            let label = format!(
                                "{shape} subgroup {subgroup} threads {threads} rows {tile} lanes {lanes} lse {with_lse}"
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
    }
}
