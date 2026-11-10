//! A recurrent state read through a segment table lands what the slot layout
//! lands, bit for bit: each family's rows and state, its slots cut along the
//! family's atom axis (heads, channels; the n-gram window and the replay
//! floor whole) into pieces scattered over a two-segment pool with garbage
//! between them, the garbage left alone. A piece offset off by a few
//! elements does not.

#![cfg(feature = "rocm")]

mod common;
#[path = "common/segments.rs"]
mod segments;

use common::{Gpu, STATE};
use dtype::Dtype;
use kernels_rocm::ssm;
use kernels_rocm::{RaggedTensor, RecurrentPool, Segments, Snapshots, Tensor};
use segments::{Bits, Cut, Lcg, Scattered, Slab, agree, bits, indptr};

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
            _ => T::of(u64::from(common::bf16(0.5 * lcg.unit()))),
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

/// Snapshots `rows` of the lanes `per_lane` counts (a CSR over lanes).
fn snapshots(gpu: &Gpu, per_lane: &[i32], rows: &[i32]) -> Snapshots {
    let mut at = vec![0i32];
    for &n in per_lane {
        at.push(at.last().unwrap() + n);
    }
    Snapshots {
        indptr: gpu.column(&at),
        rows: gpu.column(rows),
        slots: Tensor::ABSENT,
        count: rows.len() as u32,
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

/// The gated delta step on each arm it takes here: a block a (head, row)
/// over 128-wide heads, and the tiled step over 64-wide ones.
#[test]
fn the_gated_delta_step_reads_its_state_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
    for (d, pieces) in [(128u32, 2usize), (128, 4), (64, 2)] {
        let g = Delta {
            kh: 2,
            vh: 4,
            kd: d,
            vd: d,
        };
        let n = 3usize;
        let mut lcg = Lcg::seeded(0x6d6e);
        let w = g.conv_dim();
        let qkv = gpu.upload(&lcg.bf16s(n * w, 1.0), n as u32, w as u32, Dtype::Bf16);
        let gates = gpu.tensor(&g.gates(&mut lcg, n), n as u32, 2 * g.vh, Dtype::F32);
        let bank = (g.vh * g.kd * g.vd) as usize;
        let slab = heads_slab::<u16>(&mut lcg, 4, bank, 8, pieces, Dtype::Bf16);
        let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
            let mut y = gpu.zeros(n as u32, g.vh * g.vd, Dtype::F32);
            ssm::gated_delta(
                &gpu.ctx(),
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
            .expect("the step fires");
            bits::<u32>(gpu, &y)
        };
        agree(
            &gpu,
            &format!("gdn step d={d} pieces={pieces}"),
            &slab,
            &[2, -1, 3],
            Snapshots::NONE,
            &[],
            &mut fire,
            8,
        );
    }
}

/// The chunked gated delta rule: the chunk-parallel passes over 128-wide
/// heads and the token walk (both landing snapshots), and the narrow-head
/// walk.
#[test]
fn the_chunked_gated_delta_reads_its_state_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
    let lens = [70usize, 9, 33];
    let n: usize = lens.iter().sum();
    for (d, serial, snaps) in [(128u32, false, true), (128, true, true), (64, false, false)] {
        let g = Delta {
            kh: 2,
            vh: 4,
            kd: d,
            vd: d,
        };
        let mut lcg = Lcg::seeded(0x6763);
        let w = g.conv_dim();
        let qkv_t = gpu.upload(&lcg.bf16s(n * w, 1.0), n as u32, w as u32, Dtype::Bf16);
        let gates = gpu.tensor(&g.gates(&mut lcg, n), n as u32, 2 * g.vh, Dtype::F32);
        let bank = (g.vh * g.kd * g.vd) as usize;
        let slab = heads_slab::<u16>(&mut lcg, 6, bank, 0, 2, Dtype::Bf16);
        let ind = indptr(&gpu, &lens);
        // The first lane snapshots after rows 20 and 64, the last after 5.
        let (snapshots, snap_slots) = if snaps {
            (snapshots(&gpu, &[2, 0, 1], &[20, 64, 5]), vec![4, 5, 0])
        } else {
            (Snapshots::NONE, vec![])
        };
        let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
            let mut y = gpu.zeros(n as u32, g.vh * g.vd, Dtype::F32);
            let qkv = RaggedTensor {
                data: qkv_t,
                indptr: ind,
            };
            let ctx = gpu.ctx();
            let run = if serial {
                ssm::gated_delta_chunked_serial
            } else {
                ssm::gated_delta_chunked
            };
            run(
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
            .expect("the chunked rule fires");
            bits::<u32>(gpu, &y)
        };
        agree(
            &gpu,
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

/// The fused decode step reads both its states through their own segment
/// tables: the delta state cut along heads, the conv window along channels.
#[test]
fn the_fused_gated_delta_step_reads_both_states_through_their_segment_tables() {
    let Some(gpu) = Gpu::open() else { return };
    const D: u32 = 128;
    const TAPS: usize = 4;
    let (kh, vh) = (2u32, 4u32);
    let c = (2 * kh * D + vh * D) as usize;
    let zw = (vh * D) as usize;
    let n = 3usize;
    let lanes = [2i32, -1, 0];
    let mut lcg = Lcg::seeded(0xf05e);
    let qkvz = gpu.upload(
        &lcg.bf16s(n * (c + zw), 1.0),
        n as u32,
        (c + zw) as u32,
        Dtype::Bf16,
    );
    let conv_w = gpu.upload(
        &lcg.bf16s(c * TAPS, 0.5),
        c as u32,
        TAPS as u32,
        Dtype::Bf16,
    );
    let ba = gpu.upload(
        &lcg.bf16s(n * 2 * vh as usize, 1.5),
        n as u32,
        2 * vh,
        Dtype::Bf16,
    );
    let dt_bias = gpu.upload(&lcg.bf16s(vh as usize, 0.5), 1, vh, Dtype::Bf16);
    let a_log = gpu.upload(&lcg.f32s(vh as usize, 0.5), 1, vh, Dtype::F32);
    let state = heads_slab::<u16>(&mut lcg, 3, zw * D as usize, 0, 2, Dtype::Bf16);
    let conv = conv_slab(&mut lcg, 3, TAPS, c, 4);

    let fire = |state_pool: RecurrentPool, conv_pool: RecurrentPool| {
        let mut qkv = gpu.zeros(n as u32, c as u32, Dtype::Bf16);
        let mut z = gpu.zeros(n as u32, zw as u32, Dtype::Bf16);
        let mut y = gpu.zeros(n as u32, zw as u32, Dtype::F32);
        let fused = ssm::gdn_step_fused(
            &gpu.ctx(),
            qkvz,
            conv_w,
            &conv_pool,
            TAPS as u32,
            1,
            ba,
            dt_bias,
            a_log,
            &state_pool,
            kh,
            vh,
            D,
            D,
            &mut qkv,
            &mut z,
            &mut y,
        )
        .expect("the fused step fires");
        assert!(fused, "the fused step covers this shape");
        [bits::<u16>(&gpu, &qkv), bits::<u32>(&gpu, &y)].concat()
    };
    let pool = |slab: Tensor, ids: Tensor, chunk: i64, segments: Segments| RecurrentPool {
        slab,
        slot_ids: ids,
        slot_stride_elems: chunk,
        conv_slab: slab,
        conv_stride: chunk,
        segments,
        ..STATE
    };

    let flat = gpu.upload(&state.seed, 3, state.stride as u32, Dtype::Bf16);
    let conv_flat = gpu.upload(&conv.seed, 3, conv.stride as u32, Dtype::Bf16);
    let ids = gpu.column(&lanes);
    let want = fire(
        pool(flat, ids, state.stride as i64, Segments::ONE),
        pool(conv_flat, ids, conv.stride as i64, Segments::ONE),
    );
    let want_state = gpu.read_as::<u16>(&flat);
    let want_conv = gpu.read_as::<u16>(&conv_flat);

    let sc = Scattered::new(&gpu, 0x5e65, &state.seed, 3, state.stride, &state.cut);
    let cc = Scattered::new(&gpu, 0xc0c0, &conv.seed, 3, conv.stride, &conv.cut);
    let got = fire(
        pool(
            sc.slab(state.stride as u32, Dtype::Bf16),
            sc.ids(&gpu, &lanes),
            sc.chunk(),
            sc.segments(0),
        ),
        pool(
            cc.slab(conv.stride as u32, Dtype::Bf16),
            cc.ids(&gpu, &lanes),
            cc.chunk(),
            cc.segments(0),
        ),
    );
    gpu.sync();
    let (got_state, kept) = sc.gather(&gpu, &state.seed, state.stride, &state.cut);
    let (got_conv, conv_kept) = cc.gather(&gpu, &conv.seed, conv.stride, &conv.cut);
    assert!(
        got == want,
        "the scattered pools' rows are not the slot layout's"
    );
    assert!(
        got_state == want_state,
        "the scattered delta state is not the slot layout's"
    );
    assert!(
        got_conv == want_conv,
        "the scattered conv window is not the slot layout's"
    );
    assert!(
        kept && conv_kept,
        "a fire over the scattered pools wrote between their pieces"
    );
}

/// KDA's decode step and its prompt scans (the chunk-parallel passes and the
/// walk, both landing snapshots), an f32 state cut between its two heads.
#[test]
fn kda_reads_its_state_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
    const H: u32 = 2;
    const D: u32 = 128;
    let wide = (H * D) as usize;
    let lens = [21usize, 1, 7];
    let n: usize = lens.iter().sum();
    let mut lcg = Lcg::seeded(0x4b44);
    let mixed = gpu.upload(
        &lcg.bf16s(n * 3 * wide, 1.0),
        n as u32,
        3 * wide as u32,
        Dtype::Bf16,
    );
    let f = gpu.upload(
        &lcg.bf16s(n * wide, 1.0),
        n as u32,
        wide as u32,
        Dtype::Bf16,
    );
    let b = gpu.upload(&lcg.bf16s(n * H as usize, 1.0), n as u32, H, Dtype::Bf16);
    let dt_bias = gpu.upload(&lcg.f32s(wide, 0.1), 1, wide as u32, Dtype::F32);
    let a_log: Vec<f32> = (0..H).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let a_log = gpu.tensor(&a_log, 1, H, Dtype::F32);
    let slab = heads_slab::<u32>(&mut lcg, 5, (H * D * D) as usize, 0, 2, Dtype::F32);
    let ind = indptr(&gpu, &lens);
    for (step, serial) in [(true, false), (false, false), (false, true)] {
        let rows = if step { 3 } else { n };
        let (snaps, snap_slots) = if step {
            (Snapshots::NONE, vec![])
        } else {
            (snapshots(&gpu, &[1, 0, 1], &[10, 3]), vec![1, 4])
        };
        let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
            let mut y = gpu.zeros(rows as u32, wide as u32, Dtype::F32);
            let mixed_t = Tensor::new(mixed.ptr, rows as u32, 3 * wide as u32, Dtype::Bf16);
            let f_t = Tensor::new(f.ptr, rows as u32, wide as u32, Dtype::Bf16);
            let b_t = Tensor::new(b.ptr, rows as u32, H, Dtype::Bf16);
            let ctx = gpu.ctx();
            if step {
                ssm::kda(
                    &ctx, mixed_t, f_t, b_t, dt_bias, a_log, &pool, H, D, 1e-6, 0.0, &mut y,
                )
            } else {
                let data = RaggedTensor {
                    data: mixed_t,
                    indptr: ind,
                };
                let run = if serial {
                    ssm::kda_chunked_serial
                } else {
                    ssm::kda_chunked
                };
                run(
                    &ctx, data, f_t, b_t, dt_bias, a_log, &pool, H, D, 1e-6, 0.0, &mut y,
                )
            }
            .expect("KDA fires");
            bits::<u32>(gpu, &y)
        };
        agree(
            &gpu,
            &format!("kda step={step} serial={serial}"),
            &slab,
            &[3, 0, 2],
            snaps,
            &snap_slots,
            &mut fire,
            8,
        );
    }
}

/// Mamba-2's prompt scan (landing a snapshot) and decode step, an f32 state
/// cut into four runs of four heads.
#[test]
fn mamba2_reads_its_state_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
    const HEADS: usize = 16;
    const GROUPS: usize = 2;
    const P: usize = 64;
    const N: usize = 128;
    const INNER: usize = HEADS * P;
    const WIDTH: usize = INNER + 2 * GROUPS * N;
    let lens = [37usize, 5];
    let n: usize = lens.iter().sum();
    let mut lcg = Lcg::seeded(0x3a32);
    let xbc = gpu.upload(
        &lcg.bf16s(n * WIDTH, 1.0),
        n as u32,
        WIDTH as u32,
        Dtype::Bf16,
    );
    let dt = gpu.upload(
        &lcg.bf16s(n * HEADS, 1.0),
        n as u32,
        HEADS as u32,
        Dtype::Bf16,
    );
    let z = gpu.upload(
        &lcg.bf16s(n * INNER, 1.0),
        n as u32,
        INNER as u32,
        Dtype::Bf16,
    );
    let a_log: Vec<f32> = (0..HEADS).map(|_| -1.0 + 0.5 * lcg.unit()).collect();
    let m = ssm::Mamba2 {
        a_log: gpu.tensor(&a_log, 1, HEADS as u32, Dtype::F32),
        dt_bias: gpu.upload(&lcg.f32s(HEADS, 0.5), 1, HEADS as u32, Dtype::F32),
        d: gpu.upload(&lcg.f32s(HEADS, 1.0), 1, HEADS as u32, Dtype::F32),
        norm: gpu.upload(&lcg.bf16s(INNER, 1.0), 1, INNER as u32, Dtype::Bf16),
        eps: 1e-5,
        heads: HEADS as u32,
        groups: GROUPS as u32,
    };
    let slab = heads_slab::<u32>(&mut lcg, 3, HEADS * P * N, 0, 4, Dtype::F32);
    let ind = indptr(&gpu, &lens);
    for step in [false, true] {
        let rows = if step { lens.len() } else { n };
        let (snaps, snap_slots) = if step {
            (Snapshots::NONE, vec![])
        } else {
            (snapshots(&gpu, &[1, 0], &[10]), vec![1])
        };
        let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
            let mut y = gpu.zeros(rows as u32, INNER as u32, Dtype::Bf16);
            let x = Tensor::new(xbc.ptr, rows as u32, WIDTH as u32, Dtype::Bf16);
            let dt = Tensor::new(dt.ptr, rows as u32, HEADS as u32, Dtype::Bf16);
            let z = Tensor::new(z.ptr, rows as u32, INNER as u32, Dtype::Bf16);
            if step {
                ssm::mamba2(&gpu.ctx(), x, dt, z, m, &pool, N as u32, &mut y)
            } else {
                let x = RaggedTensor {
                    data: x,
                    indptr: ind,
                };
                ssm::mamba2_chunked(&gpu.ctx(), x, dt, z, m, &pool, N as u32, &mut y)
            }
            .expect("Mamba-2 fires");
            bits::<u16>(gpu, &y)
        };
        agree(
            &gpu,
            &format!("mamba2 step={step}"),
            &slab,
            &[2, 0],
            snaps,
            &snap_slots,
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
        seed: lcg.bf16s(slots * taps * c, 1.0),
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

/// The causal conv's decode update (the unrolled walk and the dilated
/// loop), its prompt arms (the eight-channel walk, and the per-channel one
/// for a piece of channels not whole eights, and for a long reach), and the
/// commit landing snapshots, a window cut along channels.
#[test]
fn the_causal_conv_reads_its_window_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
    struct Case {
        name: &'static str,
        c: usize,
        k: u32,
        dil: u32,
        pieces: usize,
        chunked: bool,
        snaps: bool,
    }
    let case = |name, c, k, dil, pieces, chunked, snaps| Case {
        name,
        c,
        k,
        dil,
        pieces,
        chunked,
        snaps,
    };
    let cases = [
        case("update", 256, 4, 1, 4, false, false),
        case("update dilated", 200, 4, 2, 2, false, false),
        case("chunked v8", 256, 4, 1, 2, true, true),
        case("chunked v8 four pieces", 256, 3, 1, 4, true, false),
        case("chunked, pieces of 100 channels", 200, 4, 1, 2, true, true),
        case("long reach", 200, 4, 12, 2, true, false),
    ];
    let lens = [45usize, 3, 17];
    for case in cases {
        let span = (case.k as usize - 1) * case.dil as usize + 1;
        let c = case.c;
        let mut lcg = Lcg::seeded(0xc0);
        let n: usize = if case.chunked { lens.iter().sum() } else { 3 };
        let x_t = gpu.upload(&lcg.bf16s(n * c, 1.0), n as u32, c as u32, Dtype::Bf16);
        let weight = gpu.upload(
            &lcg.bf16s(c * case.k as usize, 1.0),
            c as u32,
            case.k,
            Dtype::Bf16,
        );
        let slab = conv_slab(&mut lcg, 6, span, c, case.pieces);
        let ind = indptr(&gpu, &lens);
        let (snaps, snap_slots) = if case.snaps {
            (snapshots(&gpu, &[2, 0, 1], &[1, 30, 9]), vec![3, 4, 0])
        } else {
            (Snapshots::NONE, vec![])
        };
        let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
            let mut y = gpu.zeros(n as u32, c as u32, Dtype::Bf16);
            let ctx = gpu.ctx();
            if case.chunked {
                let x = RaggedTensor {
                    data: x_t,
                    indptr: ind,
                };
                ssm::causal_conv1d_chunked(&ctx, x, weight, None, &pool, case.k, case.dil, &mut y)
            } else {
                ssm::causal_conv1d(&ctx, x_t, weight, None, &pool, case.k, case.dil, &mut y)
            }
            .expect("the conv fires");
            bits::<u16>(gpu, &y)
        };
        agree(
            &gpu,
            &format!("conv {}", case.name),
            &slab,
            &[1, 5, 2],
            snaps,
            &snap_slots,
            &mut fire,
            8,
        );
    }
}

const MULTS: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xc2b2_ae3d_27d4_eb4f,
    0x1656_67b1_9e37_79f9,
];
const PRIMES: [u64; 4] = [1_000_003, 999_983, 1_000_033, 999_979];
const OFFSETS: [u64; 4] = [0, 1_000_003, 2_000_000, 3_000_000];
const EOS: u32 = 2;

/// The n-gram window (uncuttable: one piece a row) through a segment table:
/// the decode update and the chunked walk landing a snapshot.
#[test]
fn the_ngram_window_reads_its_state_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
    const STRIDE: usize = 4;
    let mut lcg = Lcg::seeded(0x9a3);
    // Cells hold an id + 1 (0: none).
    let seed: Vec<u32> = (0..5 * STRIDE)
        .map(|_| (lcg.bits() >> 40) as u32 % 50)
        .collect();
    let slab = Slab {
        seed,
        slots: 5,
        stride: STRIDE,
        dtype: Dtype::I32,
        cut: Cut::runs(1, STRIDE),
    };
    let lens = [5usize, 1, 3];
    let n: usize = lens.iter().sum();
    let tok: Vec<i32> = (0..n).map(|i| ((i * 13 + 5) % 60) as i32).collect();
    let ind = indptr(&gpu, &lens);
    for chunked in [false, true] {
        let rows = if chunked { n } else { 3 };
        let (snaps, snap_slots) = if chunked {
            (snapshots(&gpu, &[1, 0, 1], &[3, 1]), vec![1, 4])
        } else {
            (Snapshots::NONE, vec![])
        };
        let ids = gpu.column(&tok[..rows]);
        let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
            let mut out = gpu.zeros(rows as u32, 4, Dtype::I32);
            let ctx = gpu.ctx();
            if chunked {
                let x = RaggedTensor {
                    data: ids,
                    indptr: ind,
                };
                kernels_rocm::gather::ngram_ids_chunked(
                    &ctx, x, &pool, EOS, &MULTS, &PRIMES, &OFFSETS, 2, None, &mut out,
                )
            } else {
                kernels_rocm::gather::ngram_ids(
                    &ctx, ids, &pool, EOS, &MULTS, &PRIMES, &OFFSETS, 2, None, &mut out,
                )
            }
            .expect("the n-gram ids fire");
            bits::<u32>(gpu, &out)
        };
        agree(
            &gpu,
            &format!("ngram chunked={chunked}"),
            &slab,
            &[3, -1, 0],
            snaps,
            &snap_slots,
            &mut fire,
            1,
        );
    }
}

/// Bounded replay's floor (one i32 a row) through a segment table: read,
/// then marked.
#[test]
fn the_replay_floor_reads_its_state_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
    let slab = Slab {
        seed: vec![7u32, 40, 13, 99],
        slots: 4,
        stride: 1,
        dtype: Dtype::I32,
        cut: Cut::runs(1, 1),
    };
    let lens = [3usize, 2, 4];
    let n: usize = lens.iter().sum();
    let positions: Vec<i32> = (0..n as i32).map(|i| 100 + 3 * i).collect();
    let positions = RaggedTensor {
        data: gpu.column(&positions),
        indptr: indptr(&gpu, &lens),
    };
    let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
        let ctx = gpu.ctx();
        let mut read = gpu.zeros(n as u32, 1, Dtype::I32);
        let mut marked = gpu.zeros(n as u32, 1, Dtype::I32);
        kernels_rocm::attn::kv_window::floor(&ctx, positions, &pool, false, &mut read)
            .expect("the floor read fires");
        kernels_rocm::attn::kv_window::floor(&ctx, positions, &pool, true, &mut marked)
            .expect("the floor mark fires");
        [bits::<u32>(gpu, &read), bits::<u32>(gpu, &marked)].concat()
    };
    agree(
        &gpu,
        "replay floor",
        &slab,
        &[2, -1, 0],
        Snapshots::NONE,
        &[],
        &mut fire,
        1,
    );
}
