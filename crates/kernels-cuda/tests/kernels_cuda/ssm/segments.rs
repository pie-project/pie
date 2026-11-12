#![cfg(feature = "cuda")]

//! A recurrent state read through a segment table lands what the slot layout
//! lands, bit for bit: each family's rows and state, its slots cut along the
//! family's atom axis (heads, channels) into pieces scattered over a
//! two-segment pool with garbage between them, the garbage left alone. A
//! piece offset off by a few elements does not.

use crate::common::attn_ref::tuned;
use crate::common::segments::{Bits, Cut, Scattered};
use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};
use kernels_cuda::{Segments, Snapshots};

/// A fire over `pool`: its output rows, as bits.
pub type Fire<'a> = dyn FnMut(&mut Gpu, RecurrentPool) -> Vec<u64> + 'a;

/// The state a family keeps: `slots` rows `stride` elements apart, cut by
/// `cut`.
pub struct Slab<'a, T> {
    pub seed: Vec<T>,
    pub slots: usize,
    pub stride: usize,
    pub dtype: Dtype,
    pub cut: Cut<'a>,
}

fn pool(
    slab: Tensor,
    ids: Tensor,
    chunk: i64,
    segments: Segments,
    snapshots: Snapshots,
) -> RecurrentPool {
    RecurrentPool {
        slab,
        slot_ids: ids,
        slot_stride_elems: chunk,
        conv_slab: slab,
        conv_stride: chunk,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
        snapshots,
        segments,
    }
}

/// `fire` over the slot layout and over the scattered pool (lanes in slots
/// `lanes`, snapshots landing in slots `snap_slots` where `snaps` asks):
/// the same rows and state, bit for bit, and the garbage kept; with the
/// piece offset moved by `skew` elements, not.
#[allow(clippy::too_many_arguments)]
pub fn agree<T: Bits>(
    gpu: &mut Gpu,
    what: &str,
    slab: &Slab<T>,
    lanes: &[i32],
    snaps: Snapshots,
    snap_slots: &[i32],
    fire: &mut Fire,
    skew: i64,
) {
    let row = slab.stride as u32;
    let up_ids = |gpu: &mut Gpu, v: &[i32]| Tensor::new(gpu.up(v), v.len() as u32, 1, Dtype::I32);
    let with_slots = |slots: Tensor| {
        if snap_slots.is_empty() {
            snaps
        } else {
            Snapshots { slots, ..snaps }
        }
    };

    let slab_at = gpu.up(&slab.seed);
    let ids = up_ids(gpu, lanes);
    let snap_ids = with_slots(if snap_slots.is_empty() {
        Tensor::ABSENT
    } else {
        up_ids(gpu, snap_slots)
    });
    let flat = Tensor::new(slab_at, slab.slots as u32, row, slab.dtype);
    let want_rows = fire(
        gpu,
        pool(flat, ids, slab.stride as i64, Segments::ONE, snap_ids),
    );
    gpu.sync();
    let want_slab: Vec<T> = gpu.down(slab_at, slab.slots * slab.stride);

    let scatter = |gpu: &mut Gpu, skew: i64, fire: &mut Fire| {
        let sc = Scattered::new(gpu, 0x5e65, &slab.seed, slab.slots, slab.stride, &slab.cut);
        let ids = sc.ids(gpu, lanes);
        let snap_ids = with_slots(if snap_slots.is_empty() {
            Tensor::ABSENT
        } else {
            sc.ids(gpu, snap_slots)
        });
        let p = pool(
            sc.slab(row, slab.dtype),
            ids,
            sc.chunk(),
            sc.segments(skew),
            snap_ids,
        );
        let rows = fire(gpu, p);
        gpu.sync();
        let (state, kept) = sc.gather(gpu, &slab.seed, slab.stride, &slab.cut);
        (rows, state, kept)
    };

    let (rows, state, kept) = scatter(gpu, 0, fire);
    assert!(
        rows == want_rows,
        "{what}: the scattered pool's rows are not the slot layout's"
    );
    assert!(
        state == want_slab,
        "{what}: the scattered pool's state is not the slot layout's"
    );
    assert!(
        kept,
        "{what}: a fire over the scattered pool wrote between its pieces"
    );

    let (rows, state, kept) = scatter(gpu, skew, fire);
    assert!(
        rows != want_rows || state != want_slab || !kept,
        "{what}: a piece offset {skew} elements off still answers the slot layout"
    );
}

/// The raw bits of `count` elements of `T` at `at`, widened.
pub fn bits<T: Bits>(gpu: &Gpu, at: u64, count: usize) -> Vec<u64> {
    gpu.down::<T>(at, count)
        .into_iter()
        .map(Bits::wide)
        .collect()
}

fn indptr(gpu: &mut Gpu, lens: &[usize]) -> Tensor {
    let mut at = vec![0i32];
    for &len in lens {
        at.push(at.last().unwrap() + len as i32);
    }
    Tensor::new(gpu.up(&at), at.len() as u32, 1, Dtype::I32)
}

fn f32s(lcg: &mut Lcg, n: usize, scale: f32) -> Vec<u32> {
    (0..n).map(|_| (scale * lcg.unit()).to_bits()).collect()
}

/// A slot of the delta rules' state, `[heads][key][value]` in bf16 or f32,
/// cut into `pieces` runs of whole heads.
fn heads_slab<T: Bits>(
    lcg: &mut Lcg,
    slots: usize,
    bank: usize,
    pad: usize,
    pieces: usize,
    dtype: Dtype,
) -> Slab<'static, T> {
    let stride = bank + pad;
    let seed: Vec<T> = (0..slots * stride)
        .map(|_| match dtype {
            Dtype::F32 => T::of(u64::from((0.05 * lcg.unit()).to_bits())),
            _ => T::of(u64::from(crate::common::to_bf16(0.5 * lcg.unit()))),
        })
        .collect();
    Slab {
        seed,
        slots,
        stride,
        dtype,
        cut: Cut::runs(pieces, bank / pieces),
    }
}

struct Delta {
    kh: u32,
    vh: u32,
    kd: u32,
    vd: u32,
}

impl Delta {
    fn conv_dim(&self) -> usize {
        (2 * self.kh * self.kd + self.vh * self.vd) as usize
    }

    fn gates(&self, lcg: &mut Lcg, n: usize) -> Vec<f32> {
        let vh = self.vh as usize;
        let mut g = vec![0f32; n * 2 * vh];
        for r in 0..n {
            for h in 0..vh {
                g[r * 2 * vh + h] = -0.05 - 0.5 * lcg.unit().abs();
                g[r * 2 * vh + vh + h] = 0.5 + 0.4 * lcg.unit();
            }
        }
        g
    }
}

/// The gated delta step on each arm it takes here: the fused one, the
/// shared-memory one (128-wide heads, a qkv row off its 16-byte boundary),
/// and the plain one (64-wide heads, off the boundary).
#[test]
fn the_gated_delta_step_reads_its_state_through_the_segment_table() {
    for (d, misalign, pieces) in [(128u32, false, 2usize), (128, true, 4), (64, true, 2)] {
        let g = Delta {
            kh: 2,
            vh: 4,
            kd: d,
            vd: d,
        };
        let n = 3usize;
        let mut lcg = Lcg::seeded(0x6d6e);
        let mut gpu = Gpu::open();
        let w = g.conv_dim();
        let (qkv, _) = lcg.row(n * w + 8);
        let qkv_at = gpu.up(&qkv) + if misalign { 2 } else { 0 };
        let gates_at = gpu.up(&g.gates(&mut lcg, n));
        let bank = (g.vh * g.kd * g.vd) as usize;
        let slab = heads_slab::<u16>(&mut lcg, 4, bank, 8, pieces, Dtype::Bf16);
        let mut fire = |gpu: &mut Gpu, pool: RecurrentPool| {
            let y_at = gpu.zeros(n * (g.vh * g.vd) as usize * 4);
            let mut y = Tensor::new(y_at, n as u32, g.vh * g.vd, Dtype::F32);
            ssm::gated_delta(
                &gpu.ctx(),
                Tensor::new(qkv_at, n as u32, w as u32, Dtype::Bf16),
                Tensor::ABSENT,
                Tensor::new(gates_at, n as u32, 2 * g.vh, Dtype::F32),
                &pool,
                g.kh,
                g.vh,
                g.kd,
                g.vd,
                &mut y,
            )
            .expect("the step fires");
            gpu.sync();
            bits::<u32>(gpu, y_at, n * (g.vh * g.vd) as usize)
        };
        agree(
            &mut gpu,
            &format!("gdn step d={d} misaligned={misalign} pieces={pieces}"),
            &slab,
            &[2, -1, 3],
            Snapshots::NONE,
            &[],
            &mut fire,
            8,
        );
    }
}

/// The chunked gated delta rule: the split scan over 128-wide heads, the
/// serial walk landing snapshots, and the narrow-head scan.
#[test]
fn the_chunked_gated_delta_reads_its_state_through_the_segment_table() {
    let lens = [70usize, 9, 33];
    let n: usize = lens.iter().sum();
    for (d, serial, snaps) in [
        (128u32, false, false),
        (128, true, true),
        (64, false, false),
    ] {
        let g = Delta {
            kh: 2,
            vh: 4,
            kd: d,
            vd: d,
        };
        let mut lcg = Lcg::seeded(0x6763);
        let mut gpu = Gpu::open();
        let w = g.conv_dim();
        let qkv_at = gpu.up(&lcg.row(n * w).0);
        let gates_at = gpu.up(&g.gates(&mut lcg, n));
        let bank = (g.vh * g.kd * g.vd) as usize;
        let slab = heads_slab::<u16>(&mut lcg, 6, bank, 0, 2, Dtype::Bf16);
        let ind = indptr(&mut gpu, &lens);
        // The first lane snapshots after rows 20 and 64, the last after 5.
        let (snapshots, snap_slots): (Snapshots, Vec<i32>) = if snaps {
            let up =
                |gpu: &mut Gpu, v: &[i32]| Tensor::new(gpu.up(v), v.len() as u32, 1, Dtype::I32);
            (
                Snapshots {
                    indptr: up(&mut gpu, &[0, 2, 2, 3]),
                    rows: up(&mut gpu, &[20, 64, 5]),
                    slots: Tensor::ABSENT,
                    count: 3,
                },
                vec![4, 5, 0],
            )
        } else {
            (Snapshots::NONE, vec![])
        };
        let mut fire = |gpu: &mut Gpu, pool: RecurrentPool| {
            let y_at = gpu.zeros(n * (g.vh * g.vd) as usize * 4);
            let mut y = Tensor::new(y_at, n as u32, g.vh * g.vd, Dtype::F32);
            let qkv = RaggedTensor {
                data: Tensor::new(qkv_at, n as u32, w as u32, Dtype::Bf16),
                indptr: ind,
            };
            let gates = Tensor::new(gates_at, n as u32, 2 * g.vh, Dtype::F32);
            let ctx = gpu.ctx();
            if serial {
                ssm::gated_delta_chunked_serial(
                    &ctx,
                    qkv,
                    Tensor::ABSENT,
                    gates,
                    &pool,
                    g.kh,
                    g.vh,
                    g.kd,
                    g.vd,
                    &mut y,
                )
            } else {
                ssm::gated_delta_chunked(
                    &ctx,
                    qkv,
                    Tensor::ABSENT,
                    gates,
                    &pool,
                    g.kh,
                    g.vh,
                    g.kd,
                    g.vd,
                    &mut y,
                )
            }
            .expect("the chunked rule fires");
            gpu.sync();
            bits::<u32>(gpu, y_at, n * (g.vh * g.vd) as usize)
        };
        agree(
            &mut gpu,
            &format!("gdn chunked d={d} serial={serial} snapshots={snaps}"),
            &slab,
            &[1, 3, 2],
            snapshots,
            &snap_slots,
            &mut fire,
            8,
        );
    }
}

/// KDA's decode step and its prompt scans (the tensor-core scan and the
/// walk), an f32 state cut between its two heads.
#[test]
fn kda_reads_its_state_through_the_segment_table() {
    const H: u32 = 2;
    const D: u32 = 128;
    let wide = (H * D) as usize;
    let lens = [21usize, 1, 7];
    let n: usize = lens.iter().sum();
    let mut lcg = Lcg::seeded(0x4b44);
    let mut gpu = Gpu::open();
    let mixed = gpu.up(&lcg.row(n * 3 * wide).0);
    let f = gpu.up(&lcg.row(n * wide).0);
    let b = gpu.up(&lcg.row(n * H as usize).0);
    let dt_bias = gpu.up(&f32s(&mut lcg, wide, 0.1));
    let a_log: Vec<u32> = (0..H)
        .map(|_| (-0.5 + 0.2 * lcg.unit()).to_bits())
        .collect();
    let a_log = gpu.up(&a_log);
    let slab = heads_slab::<u32>(&mut lcg, 4, (H * D * D) as usize, 0, 2, Dtype::F32);
    let ind = indptr(&mut gpu, &lens);
    for (step, mma) in [(true, false), (false, true), (false, false)] {
        let rows = if step { 3 } else { n };
        let mut fire = |gpu: &mut Gpu, pool: RecurrentPool| {
            let y_at = gpu.zeros(rows * wide * 4);
            let mut y = Tensor::new(y_at, rows as u32, wide as u32, Dtype::F32);
            let mixed_t = Tensor::new(mixed, rows as u32, 3 * wide as u32, Dtype::Bf16);
            let f_t = Tensor::new(f, rows as u32, wide as u32, Dtype::Bf16);
            let b_t = Tensor::new(b, rows as u32, H, Dtype::Bf16);
            let dt_t = Tensor::new(dt_bias, 1, wide as u32, Dtype::F32);
            let a_t = Tensor::new(a_log, 1, H, Dtype::F32);
            let ctx = gpu.ctx().with_tuning(tuned(|t| t.ssm_kda_chunked = mma));
            if step {
                ssm::kda(
                    &ctx, mixed_t, f_t, b_t, dt_t, a_t, &pool, H, D, 1e-6, 0.0, &mut y,
                )
            } else {
                let data = RaggedTensor {
                    data: mixed_t,
                    indptr: ind,
                };
                ssm::kda_chunked(
                    &ctx, data, f_t, b_t, dt_t, a_t, &pool, H, D, 1e-6, 0.0, &mut y,
                )
            }
            .expect("KDA fires");
            gpu.sync();
            bits::<u32>(gpu, y_at, rows * wide)
        };
        agree(
            &mut gpu,
            &format!("kda step={step} mma={mma}"),
            &slab,
            &[3, 0, 2],
            Snapshots::NONE,
            &[],
            &mut fire,
            8,
        );
    }
}

/// Mamba-2's prompt scan and decode step, an f32 state cut into four runs
/// of four heads.
#[test]
fn mamba2_reads_its_state_through_the_segment_table() {
    const HEADS: usize = 16;
    const GROUPS: usize = 2;
    const P: usize = 64;
    const N: usize = 128;
    const INNER: usize = HEADS * P;
    const WIDTH: usize = INNER + 2 * GROUPS * N;
    let lens = [37usize, 5];
    let n: usize = lens.iter().sum();
    let mut lcg = Lcg::seeded(0x3a32);
    let mut gpu = Gpu::open();
    let xbc = gpu.up(&lcg.row(n * WIDTH).0);
    let dt = gpu.up(&lcg.row(n * HEADS).0);
    let z = gpu.up(&lcg.row(n * INNER).0);
    let a_log: Vec<u32> = (0..HEADS)
        .map(|_| (-1.0 + 0.5 * lcg.unit()).to_bits())
        .collect();
    let m = ssm::Mamba2 {
        a_log: Tensor::new(gpu.up(&a_log), 1, HEADS as u32, Dtype::F32),
        dt_bias: Tensor::new(
            gpu.up(&f32s(&mut lcg, HEADS, 0.5)),
            1,
            HEADS as u32,
            Dtype::F32,
        ),
        d: Tensor::new(
            gpu.up(&f32s(&mut lcg, HEADS, 1.0)),
            1,
            HEADS as u32,
            Dtype::F32,
        ),
        norm: Tensor::new(gpu.up(&lcg.row(INNER).0), 1, INNER as u32, Dtype::Bf16),
        eps: 1e-5,
        heads: HEADS as u32,
        groups: GROUPS as u32,
    };
    let slab = heads_slab::<u32>(&mut lcg, 3, HEADS * P * N, 0, 4, Dtype::F32);
    let ind = indptr(&mut gpu, &lens);
    for step in [false, true] {
        let rows = if step { lens.len() } else { n };
        let mut fire = |gpu: &mut Gpu, pool: RecurrentPool| {
            let y_at = gpu.zeros(rows * INNER * 2);
            let mut y = Tensor::new(y_at, rows as u32, INNER as u32, Dtype::Bf16);
            let x = Tensor::new(xbc, rows as u32, WIDTH as u32, Dtype::Bf16);
            let dt = Tensor::new(dt, rows as u32, HEADS as u32, Dtype::Bf16);
            let z = Tensor::new(z, rows as u32, INNER as u32, Dtype::Bf16);
            if step {
                ssm::mamba2(&gpu.ctx(), x, dt, z, m, &pool, N as u32, &mut y)
            } else {
                let x = RaggedTensor {
                    data: x,
                    indptr: ind,
                };
                ssm::mamba2_chunked(&gpu.ctx(), x, dt, z, m, &pool, None, N as u32, &mut y)
            }
            .expect("Mamba-2 fires");
            gpu.sync();
            bits::<u16>(gpu, y_at, rows * INNER)
        };
        agree(
            &mut gpu,
            &format!("mamba2 step={step}"),
            &slab,
            &[2, 0],
            Snapshots::NONE,
            &[],
            &mut fire,
            8,
        );
    }
}

/// A conv window `[taps][C]` cut into `pieces` channel runs: piece `p`
/// holds `[taps][C / pieces]`.
fn conv_slab(
    lcg: &mut Lcg,
    slots: usize,
    taps: usize,
    c: usize,
    pieces: usize,
) -> Slab<'static, u16> {
    let cs = c / pieces;
    let at: &'static dyn Fn(usize, usize) -> usize =
        Box::leak(Box::new(move |p: usize, i: usize| {
            (i / cs) * c + p * cs + i % cs
        }));
    Slab {
        seed: lcg.row(slots * taps * c).0,
        slots,
        stride: taps * c,
        dtype: Dtype::Bf16,
        cut: Cut {
            pieces,
            piece: taps * cs,
            at,
        },
    }
}

/// The causal conv's decode update (vector and scalar), its prompt arms
/// (the row tiles, vector and not, and the per-channel walk of a long
/// reach), and the commit landing snapshots, a window cut along channels.
#[test]
fn the_causal_conv_reads_its_window_through_the_segment_table() {
    struct Case {
        name: &'static str,
        c: usize,
        k: u32,
        dil: u32,
        pieces: usize,
        chunked: bool,
        vec4: bool,
        snaps: bool,
    }
    let cases = [
        Case {
            name: "update vec8",
            c: 256,
            k: 4,
            dil: 1,
            pieces: 4,
            chunked: false,
            vec4: false,
            snaps: false,
        },
        Case {
            name: "update scalar",
            c: 200,
            k: 4,
            dil: 2,
            pieces: 2,
            chunked: false,
            vec4: false,
            snaps: false,
        },
        Case {
            name: "row tile vec8",
            c: 256,
            k: 4,
            dil: 1,
            pieces: 2,
            chunked: true,
            vec4: false,
            snaps: true,
        },
        Case {
            name: "row tile vec4",
            c: 256,
            k: 4,
            dil: 1,
            pieces: 4,
            chunked: true,
            vec4: true,
            snaps: false,
        },
        Case {
            name: "row tile",
            c: 200,
            k: 4,
            dil: 1,
            pieces: 2,
            chunked: true,
            vec4: false,
            snaps: true,
        },
        Case {
            name: "long reach",
            c: 200,
            k: 4,
            dil: 12,
            pieces: 2,
            chunked: true,
            vec4: false,
            snaps: false,
        },
    ];
    let lens = [45usize, 3, 17];
    for case in cases {
        let span = (case.k as usize - 1) * case.dil as usize + 1;
        let c = case.c;
        let mut lcg = Lcg::seeded(0xc0);
        let mut gpu = Gpu::open();
        let n: usize = if case.chunked { lens.iter().sum() } else { 3 };
        let x_at = gpu.up(&lcg.row(n * c).0);
        let w_at = gpu.up(&lcg.row(c * case.k as usize).0);
        let slab = conv_slab(&mut lcg, 6, span, c, case.pieces);
        let ind = indptr(&mut gpu, &lens);
        let up = |gpu: &mut Gpu, v: &[i32]| Tensor::new(gpu.up(v), v.len() as u32, 1, Dtype::I32);
        let (snapshots, snap_slots) = if case.snaps {
            (
                Snapshots {
                    indptr: up(&mut gpu, &[0, 2, 2, 3]),
                    rows: up(&mut gpu, &[1, 30, 9]),
                    slots: Tensor::ABSENT,
                    count: 3,
                },
                vec![3, 4, 0],
            )
        } else {
            (Snapshots::NONE, vec![])
        };
        let mut fire = |gpu: &mut Gpu, pool: RecurrentPool| {
            let y_at = gpu.zeros(n * c * 2);
            let mut y = Tensor::new(y_at, n as u32, c as u32, Dtype::Bf16);
            let x = Tensor::new(x_at, n as u32, c as u32, Dtype::Bf16);
            let weight = Tensor::new(w_at, c as u32, case.k, Dtype::Bf16);
            let ctx = gpu
                .ctx()
                .with_tuning(tuned(|t| t.ssm_conv_vec4 = case.vec4));
            if case.chunked {
                let x = RaggedTensor {
                    data: x,
                    indptr: ind,
                };
                ssm::causal_conv1d_chunked(&ctx, x, weight, None, &pool, case.k, case.dil, &mut y)
            } else {
                ssm::causal_conv1d(&ctx, x, weight, None, &pool, case.k, case.dil, &mut y)
            }
            .expect("the conv fires");
            gpu.sync();
            bits::<u16>(gpu, y_at, n * c)
        };
        agree(
            &mut gpu,
            &format!("conv {}", case.name),
            &slab,
            &[1, 5, 2],
            snapshots,
            &snap_slots,
            &mut fire,
            8,
        );
    }
}
