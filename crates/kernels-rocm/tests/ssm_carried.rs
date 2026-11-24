#![cfg(feature = "rocm")]

//! A decode step that reads its lane's buffered rows as a linear attention
//! from the committed state (`ssm::Carry`) answers what the plain step
//! answers folding each row as it lands: each row's output, and the state
//! where it folds the rows it commits.

mod common;

use common::{Gpu, from_bf16};
use dtype::Dtype;
use kernels_rocm::ssm::{self, Carry};
use kernels_rocm::tensor::{RecurrentPool, Tensor};

/// Rows a page of the test's buffer holds.
const PAGE: u32 = 16;

/// One page of buffered planes, `bytes` a row each, laid one after another.
struct Page {
    base: u64,
    offs: Vec<u64>,
    strides: Vec<u64>,
}

impl Page {
    /// The page, plane `fill.0`'s rows from `fill.1` (bf16 words, a row
    /// after another), the rest zero.
    fn new(gpu: &mut Gpu, bytes: &[u64], fill: Option<(usize, &[u16])>) -> Self {
        let strides: Vec<u64> = bytes.iter().map(|b| b.next_multiple_of(16)).collect();
        let mut offs = Vec::new();
        let mut at = 0u64;
        for s in &strides {
            offs.push(at);
            at += s * u64::from(PAGE);
        }
        let mut words = vec![0u16; at as usize / 2];
        if let Some((plane, rows)) = fill {
            let width = bytes[plane] as usize / 2;
            for (j, chunk) in rows.chunks(width).enumerate() {
                let o = (offs[plane] + j as u64 * strides[plane]) as usize / 2;
                words[o..o + width].copy_from_slice(chunk);
            }
        }
        Page {
            base: gpu.up(&words),
            offs,
            strides,
        }
    }

    fn row(&self, plane: usize, tok: u32) -> u64 {
        self.base + self.offs[plane] + u64::from(tok) * self.strides[plane]
    }

    /// The tables of one lane landing its own row at `at`, carrying the
    /// `replay` rows before it (tokens `0..replay`), committing `commit` rows
    /// of its run.
    fn carry(&self, gpu: &mut Gpu, at: u32, replay: u32, commit: i32) -> Carry {
        let mut carry = Carry {
            lanes: gpu.up(&[at, replay, 0u32, 0, 0]),
            pages: gpu.up(&[self.base]),
            // Its rows as two spans, so a run is walked across spans.
            replay_spans: gpu.up(&[0u32, replay / 2, replay / 2, replay - replay / 2]),
            commit_len: gpu.up(&[commit]),
            page_tokens: PAGE,
            ..Carry::NONE
        };
        for (p, (&off, &stride)) in self.offs.iter().zip(&self.strides).enumerate() {
            carry.plane_off[p] = off;
            carry.plane_stride[p] = stride;
        }
        carry
    }
}

fn pool(slab: u64, stride: usize, slots: u64) -> RecurrentPool {
    RecurrentPool {
        slab: Tensor::new(slab, 1, stride as u32, Dtype::F32),
        slot_ids: Tensor::new(slots, 1, 1, Dtype::I32),
        slot_stride_elems: stride as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
        segments: kernels_rocm::Segments::ONE,
        snapshots: kernels_rocm::Snapshots::NONE,
    }
}

fn near(got: &[f32], want: &[f32], tol: f32, what: &str) {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let off = got
        .iter()
        .zip(want)
        .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(off <= tol * scale.max(1e-6), "{what}: off by {off} of {scale}");
}

/// KDA: twelve decodes of one lane, each carrying every row before it
/// unfolded, answer the plain step's rows; the last commits a prefix or the
/// whole run and leaves the plain step's state there; the chunked tail
/// lands the corrected values the decodes landed.
#[test]
fn a_kda_decode_reads_its_buffered_rows_as_the_step_folds_them() {
    const H: u32 = 2;
    const D: u32 = 128;
    const T: usize = 12;
    let wide = (H * D) as usize;
    let stride = (H * D * D) as usize;
    let Some(mut gpu) = Gpu::open() else { return };
    let mut lcg = Lcg::seeded(0x6ca);
    let (mixed_raw, _) = lcg.row(T * 3 * wide);
    let mixed = gpu.up(&mixed_raw);
    let f = gpu.up(&lcg.row(T * wide).0);
    let b = gpu.up(&lcg.row(T * H as usize).0);
    let dt_bias: Vec<f32> = (0..wide).map(|_| 0.1 * lcg.unit()).collect();
    let a_log: Vec<f32> = (0..H).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let (dt_bias, a_log) = (gpu.up(&dt_bias), gpu.up(&a_log));
    let seed: Vec<f32> = (0..stride).map(|_| 0.05 * lcg.unit()).collect();
    let slots = gpu.up(&[0i32]);
    let row = |base: u64, j: usize, width: usize, elem: usize| base + (j * width * elem) as u64;
    let dt_t = Tensor::new(dt_bias, 1, wide as u32, Dtype::F32);
    // Planes: forget, beta, keys, corrected values.
    let planes = [wide as u64 * 2, u64::from(H) * 2, wide as u64 * 2, wide as u64 * 4];
    let a_t = Tensor::new(a_log, 1, H, Dtype::F32);
    for floor in [0.0f32, -5.0] {
        // The plain step, a row a fire, folding each.
        let slab = gpu.up(&seed);
        let y = gpu.up(&vec![0u8; T * wide * 4]);
        let mut want_slabs = Vec::new();
        for j in 0..T {
            gpu.laid(&gpu.ctx(), |ctx| {
                ssm::kda(
                    ctx,
                    Tensor::new(row(mixed, j, 3 * wide, 2), 1, 3 * wide as u32, Dtype::Bf16),
                    Tensor::new(row(f, j, wide, 2), 1, wide as u32, Dtype::Bf16),
                    Tensor::new(row(b, j, H as usize, 2), 1, H, Dtype::Bf16),
                    dt_t,
                    a_t,
                    &pool(slab, stride, slots),
                    H,
                    D,
                    1e-6,
                    floor,
                    &mut Tensor::new(row(y, j, wide, 4), 1, wide as u32, Dtype::F32),
                )
            })
            .expect("the KDA step fires");
            gpu.sync();
            want_slabs.push(gpu.down::<f32>(slab, stride));
        }
        let want_y: Vec<f32> = gpu.down(y, T * wide);

        for commit in [T as i32 - 4, T as i32] {
            let keys: Vec<u16> = (0..T)
                .flat_map(|j| mixed_raw[j * 3 * wide + wide..j * 3 * wide + 2 * wide].to_vec())
                .collect();
            let page = Page::new(&mut gpu, &planes, Some((2, &keys)));
            let slab = gpu.up(&seed);
            let y = gpu.up(&vec![0u8; T * wide * 4]);
            for j in 0..T {
                let c = if j + 1 == T { commit } else { 0 };
                let carry = page.carry(&mut gpu, j as u32, j as u32, c);
                ssm::kda_carried(
                    &gpu.ctx(),
                    Tensor::new(row(mixed, j, 3 * wide, 2), 1, 3 * wide as u32, Dtype::Bf16),
                    Tensor::new(row(f, j, wide, 2), 1, wide as u32, Dtype::Bf16),
                    Tensor::new(row(b, j, H as usize, 2), 1, H, Dtype::Bf16),
                    dt_t,
                    a_t,
                    &pool(slab, stride, slots),
                    H,
                    D,
                    1e-6,
                    floor,
                    &mut Tensor::new(row(y, j, wide, 4), 1, wide as u32, Dtype::F32),
                    &carry,
                )
                .expect("the carried KDA step fires");
                gpu.sync();
                if j + 1 < T {
                    assert_eq!(gpu.down::<f32>(slab, stride), seed, "a decode committing nothing keeps the state");
                }
            }
            near(&gpu.down::<f32>(y, T * wide), &want_y, 2e-3, "the carried decodes' rows");
            near(
                &gpu.down::<f32>(slab, stride),
                &want_slabs[commit as usize - 1],
                2e-3,
                "the state folded through the commit",
            );

            // A beam group's fold of its common first rows, from the kept
            // planes, leaves the plain step's state there; a group folding
            // none beside it touches nothing.
            for n in [5u32, T as u32] {
                let slab = gpu.up(&seed);
                let carry = page.carry(&mut gpu, 0, n, 0);
                let folds = gpu.up(&[0u32, 0, n, 0, 0, 0]);
                ssm::kda_fold(
                    &gpu.ctx(),
                    &pool(slab, stride, slots),
                    H,
                    D,
                    1e-6,
                    floor,
                    dt_t,
                    a_t,
                    &carry,
                    Tensor::new(folds, 2, 3, Dtype::U32),
                    2,
                )
                .expect("the KDA group fold fires");
                gpu.sync();
                near(
                    &gpu.down::<f32>(slab, stride),
                    &want_slabs[n as usize - 1],
                    2e-3,
                    &format!("the state a group folds through {n} rows"),
                );
            }

            // The chunked tail from the state folded through `c` rows lands
            // the keys and corrected values the decodes landed.
            let c = 5usize;
            let tail = Page::new(&mut gpu, &planes, None);
            let slab = gpu.up(&want_slabs[c - 1]);
            let carry = tail.carry(&mut gpu, 0, 0, c as i32);
            let indptr = gpu.up(&[0u32, T as u32]);
            ssm::kda_tail(
                &gpu.ctx(),
                Tensor::new(mixed, T as u32, 3 * wide as u32, Dtype::Bf16),
                Tensor::new(f, T as u32, wide as u32, Dtype::Bf16),
                Tensor::new(b, T as u32, H, Dtype::Bf16),
                dt_t,
                a_t,
                &pool(slab, stride, slots),
                Tensor::new(indptr, 2, 1, Dtype::U32),
                false,
                H,
                D,
                1e-6,
                floor,
                &carry,
            )
            .expect("the KDA tail fires");
            gpu.sync();
            for j in c..T {
                let got: Vec<f32> = gpu.down(tail.row(3, j as u32), wide);
                let want: Vec<f32> = gpu.down(page.row(3, j as u32), wide);
                near(&got, &want, 2e-3, &format!("row {j}'s corrected value"));
                let got: Vec<u16> = gpu.down(tail.row(2, j as u32), wide);
                let want: Vec<u16> = gpu.down(page.row(2, j as u32), wide);
                assert_eq!(got, want, "row {j}'s keys");
            }
            assert_eq!(gpu.down::<f32>(slab, stride), want_slabs[c - 1], "the tail keeps the state");
        }
    }
}

/// Mamba-2: twelve decodes of one lane, each carrying every row before it
/// unfolded, answer the plain step's rows; the last commits a prefix or the
/// whole run and leaves the plain step's state there.
#[test]
fn a_mamba2_decode_reads_its_buffered_rows_as_the_step_folds_them() {
    const HEADS: usize = 4;
    const GROUPS: usize = 2;
    const P: usize = 64;
    const N: usize = 128;
    const INNER: usize = HEADS * P;
    const WIDTH: usize = INNER + 2 * GROUPS * N;
    const XB: usize = INNER + GROUPS * N;
    const STRIDE: usize = HEADS * P * N;
    const T: usize = 12;
    let Some(mut gpu) = Gpu::open() else { return };
    let mut lcg = Lcg::seeded(0x3a7);
    let (xbc_raw, _) = lcg.row(T * WIDTH);
    let (_, dt) = lcg.row(T * HEADS);
    let dt_raw: Vec<u16> = dt.iter().map(|v| to_bf16(v - 2.0)).collect();
    let (z_raw, _) = lcg.row(T * INNER);
    let (norm_raw, _) = lcg.row(INNER);
    let a_log: Vec<f32> = (0..HEADS).map(|_| 0.5 * lcg.unit()).collect();
    let dt_bias: Vec<f32> = (0..HEADS).map(|_| 0.5 * lcg.unit()).collect();
    let d: Vec<f32> = (0..HEADS).map(|_| lcg.unit()).collect();
    let seed: Vec<f32> = (0..STRIDE).map(|_| 0.1 * lcg.unit()).collect();
    let xbc = gpu.up(&xbc_raw);
    let dt_at = gpu.up(&dt_raw);
    let z = gpu.up(&z_raw);
    let m = ssm::Mamba2 {
        a_log: Tensor::new(gpu.up(&a_log), 1, HEADS as u32, Dtype::F32),
        dt_bias: Tensor::new(gpu.up(&dt_bias), 1, HEADS as u32, Dtype::F32),
        d: Tensor::new(gpu.up(&d), 1, HEADS as u32, Dtype::F32),
        norm: Tensor::new(gpu.up(&norm_raw), 1, INNER as u32, Dtype::Bf16),
        eps: 1e-5,
        heads: HEADS as u32,
        groups: GROUPS as u32,
    };
    let slots = gpu.up(&[0i32]);
    let row = |base: u64, j: usize, width: usize| base + (j * width * 2) as u64;
    let step = |gpu: &mut Gpu, slab: u64, y: u64, j: usize, carry: Option<&Carry>| {
        let args = (
            Tensor::new(row(xbc, j, WIDTH), 1, WIDTH as u32, Dtype::Bf16),
            Tensor::new(row(dt_at, j, HEADS), 1, HEADS as u32, Dtype::Bf16),
            Tensor::new(row(z, j, INNER), 1, INNER as u32, Dtype::Bf16),
        );
        let mut out = Tensor::new(row(y, j, INNER), 1, INNER as u32, Dtype::Bf16);
        let pool = pool(slab, STRIDE, slots);
        match carry {
            None => ssm::mamba2(&gpu.ctx(), args.0, args.1, args.2, m, &pool, N as u32, &mut out),
            Some(carry) => ssm::mamba2_carried(
                &gpu.ctx(),
                args.0,
                args.1,
                args.2,
                m,
                &pool,
                N as u32,
                &mut out,
                carry,
            ),
        }
        .expect("the Mamba-2 step fires");
        gpu.sync();
    };
    let slab = gpu.up(&seed);
    let y = gpu.up(&vec![0u8; T * INNER * 2]);
    let mut want_slabs = Vec::new();
    for j in 0..T {
        step(&mut gpu, slab, y, j, None);
        want_slabs.push(gpu.down::<f32>(slab, STRIDE));
    }
    let want_y: Vec<f32> = gpu.down::<u16>(y, T * INNER).into_iter().map(from_bf16).collect();
    for commit in [T as i32 - 4, T as i32] {
        // Planes: step sizes, `x | B`.
        let page = Page::new(&mut gpu, &[HEADS as u64 * 2, XB as u64 * 2], None);
        let indptr = gpu.up(&[0u32, T as u32]);
        let lands = page.carry(&mut gpu, 0, 0, 0);
        ssm::mamba2_tail(
            &gpu.ctx(),
            Tensor::new(xbc, T as u32, WIDTH as u32, Dtype::Bf16),
            m,
            &pool(slab, STRIDE, slots),
            Tensor::new(indptr, 2, 1, Dtype::U32),
            false,
            N as u32,
            &lands,
        )
        .expect("the Mamba-2 tail fires");
        gpu.sync();
        for j in 0..T {
            let got: Vec<u16> = gpu.down(page.row(1, j as u32), XB);
            assert_eq!(got, xbc_raw[j * WIDTH..j * WIDTH + XB].to_vec(), "row {j}'s `x | B`");
        }
        let slab = gpu.up(&seed);
        let y = gpu.up(&vec![0u8; T * INNER * 2]);
        for j in 0..T {
            let c = if j + 1 == T { commit } else { 0 };
            let carry = page.carry(&mut gpu, j as u32, j as u32, c);
            step(&mut gpu, slab, y, j, Some(&carry));
            if j + 1 < T {
                assert_eq!(gpu.down::<f32>(slab, STRIDE), seed, "a decode committing nothing keeps the state");
            }
        }
        let got: Vec<f32> = gpu.down::<u16>(y, T * INNER).into_iter().map(from_bf16).collect();
        near(&got, &want_y, 2e-2, "the carried decodes' rows");
        near(
            &gpu.down::<f32>(slab, STRIDE),
            &want_slabs[commit as usize - 1],
            1e-4,
            "the state folded through the commit",
        );

        // A beam group's fold of its common first rows, from the kept
        // planes, leaves the plain step's state there; a group folding none
        // beside it touches nothing.
        for n in [5u32, T as u32] {
            let slab = gpu.up(&seed);
            let carry = page.carry(&mut gpu, 0, n, 0);
            let folds = gpu.up(&[0u32, 0, n, 0, 0, 0]);
            ssm::mamba2_fold(
                &gpu.ctx(),
                &pool(slab, STRIDE, slots),
                m,
                &carry,
                Tensor::new(folds, 2, 3, Dtype::U32),
                2,
            )
            .expect("the Mamba-2 group fold fires");
            gpu.sync();
            near(
                &gpu.down::<f32>(slab, STRIDE),
                &want_slabs[n as usize - 1],
                1e-4,
                &format!("the state a group folds through {n} rows"),
            );
        }
    }
}

/// `kernels-cuda`'s test helpers this suite reads.
#[must_use]
pub fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(rounding) >> 16) as u16
}


pub struct Lcg(u64);

impl Lcg {
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 40) as u32;
        #[allow(clippy::cast_precision_loss)]
        let raw = (bits as f32 / 8_388_608.0) - 1.0;
        from_bf16(to_bf16(raw))
    }

    pub fn row(&mut self, count: usize) -> (Vec<u16>, Vec<f32>) {
        let mut raw = Vec::with_capacity(count);
        let mut exact = Vec::with_capacity(count);
        for _ in 0..count {
            let value = self.unit();
            raw.push(to_bf16(value));
            exact.push(value);
        }
        (raw, exact)
    }
}
