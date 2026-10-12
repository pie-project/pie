#![cfg(target_vendor = "apple")]

//! A bank's row stride is its own field now, so the leading columns of a
//! wider bank are a view: the same planes read narrower at the wide stride.
//! Every quantized matmul path — one row, a few rows folded, tiled, precast,
//! split-k — must read that view exactly as it reads a bank copied out of
//! those columns, since the Neural Engine split hands the GPU such a view of
//! `down` in place of #837's host copy.
//!
//! The other cut a split needs is of the output: the leading `KEEP` rows of
//! the bank landed in the leading `KEEP` columns of a full-width `y`, the
//! rest of each row untouched for the other engine to fill.

use std::cell::RefCell;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

/// The whole bank: `N` output rows over `K_FULL` columns in groups of 64.
const N: u32 = 256;
const K_FULL: u32 = 2048;
/// The view keeps the leading `K_CUT` columns.
const K_CUT: u32 = 1024;
const GROUP: u32 = 64;
/// The output columns the GPU keeps in the in-place test.
const KEEP: u32 = 128;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

/// The two tests here run on two contexts at once, and a small shared
/// buffer of each shares 16 KB pages with its neighbours: with `y` sized
/// to its bytes, one run in seven read its first rows back as zero after
/// the kernel had written them — a neighbouring allocation on the other
/// thread wiping the page — while `y` on whole pages never did, nor did a
/// second process on the GPU. So each output takes whole pages.
fn whole_pages(bytes: u64) -> u64 {
    bytes.next_multiple_of(16384)
}

fn bound(device: &Context, handles: &Handles, bytes: &[u8], keep: &mut Vec<Buffer>) -> u32 {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("a buffer");
    buffer.write(0, bytes).expect("written");
    let handle = handles.bind(&buffer, 0, buffer.bytes()).expect("a handle");
    keep.push(buffer);
    handle
}

/// Row-major planes of the whole bank, and the same cut to `K_CUT` columns.
struct Planes {
    codes: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

fn planes(rng: &mut Lcg, k: u32, from: Option<&Planes>) -> Planes {
    let groups = (k / GROUP) as usize;
    match from {
        None => Planes {
            codes: (0..(N * k / 2)).map(|_| rng.next() as u8).collect(),
            scales: (0..N as usize * groups)
                .flat_map(|_| bf16(0.01 + rng.unit().abs() * 0.05).to_le_bytes())
                .collect(),
            biases: (0..N as usize * groups)
                .flat_map(|_| bf16(rng.unit() * 0.1).to_le_bytes())
                .collect(),
        },
        Some(whole) => {
            let (row_in, row_out) = ((K_FULL / 2) as usize, (k / 2) as usize);
            let (g_in, g_out) = ((K_FULL / GROUP) as usize * 2, groups * 2);
            Planes {
                codes: whole
                    .codes
                    .chunks_exact(row_in)
                    .flat_map(|row| row[..row_out].to_vec())
                    .collect(),
                scales: whole
                    .scales
                    .chunks_exact(g_in)
                    .flat_map(|row| row[..g_out].to_vec())
                    .collect(),
                biases: whole
                    .biases
                    .chunks_exact(g_in)
                    .flat_map(|row| row[..g_out].to_vec())
                    .collect(),
            }
        }
    }
}

fn bank(
    device: &Context,
    handles: &Handles,
    p: &Planes,
    width: u32,
    ld: u32,
    keep: &mut Vec<Buffer>,
) -> Bank {
    let groups = width / GROUP;
    Bank {
        codes: Tensor::new(
            bound(device, handles, &p.codes, keep),
            N,
            width,
            Dtype::U4g64,
        ),
        mpp_codes: None,
        scales: Tensor::new(
            bound(device, handles, &p.scales, keep),
            N,
            groups,
            Dtype::Bf16,
        ),
        biases: Some(Tensor::new(
            bound(device, handles, &p.biases, keep),
            N,
            groups,
            Dtype::Bf16,
        )),
        group: GROUP,
        bits: 4,
        ld,
    }
}

#[test]
fn a_column_cut_of_a_bank_multiplies_as_its_copy() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut rng = Lcg(0x5eed);
    let whole = planes(&mut rng, K_FULL, None);
    let cut = planes(&mut rng, K_CUT, Some(&whole));
    let mut keep = Vec::new();
    let view = bank(&device, &handles, &whole, K_CUT, K_FULL, &mut keep);
    let copy = bank(&device, &handles, &cut, K_CUT, K_CUT, &mut keep);

    // Scratch the dispatcher may ask for: a precast plane and split-k
    // partials, sized to whatever it wants, kept until the frame lands.
    let scratch_keep: RefCell<Vec<Buffer>> = RefCell::new(Vec::new());
    let plane = |rows: u32, width: u32, dtype: Dtype| -> Option<Tensor> {
        let bytes = u64::from(rows) * u64::from(width) * dtype.bits() / 8;
        let buffer = Buffer::zeroed(&device, bytes).ok()?;
        let handle = handles.bind(&buffer, 0, bytes).ok()?;
        scratch_keep.borrow_mut().push(buffer);
        Some(Tensor::new(handle, rows, width, dtype))
    };
    let precast = |rows, contraction| plane(rows, contraction, Dtype::F16);
    let partials = |rows, width| plane(rows, width, Dtype::F32);

    for m in [1u32, 4, 17, 64, 640] {
        let x: Vec<u8> = (0..m * K_CUT)
            .flat_map(|_| bf16(rng.unit()).to_le_bytes())
            .collect();
        let mut rows = Vec::new();
        let act = Tensor::new(
            bound(&device, &handles, &x, &mut rows),
            m,
            K_CUT,
            Dtype::Bf16,
        );
        let mut out = Vec::new();
        for w in [view, copy] {
            let y_bytes = u64::from(m) * u64::from(N) * 2;
            let y_buffer = Buffer::zeroed(&device, whole_pages(y_bytes)).expect("y");
            let y = Tensor::new(
                handles.bind(&y_buffer, 0, y_bytes).expect("y"),
                m,
                N,
                Dtype::Bf16,
            );
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            quant::matmul(
                &sink,
                act,
                w,
                y,
                quant::Scratch {
                    precast: &precast,
                    partials: &partials,
                },
                m.div_ceil(32) * 32,
            )
            .expect("the matmul launches");
            frame.commit().expect("the frame lands");
            out.push(handles.read(y.buf, y_bytes).expect("y read back"));
            scratch_keep.borrow_mut().clear();
            drop(y_buffer);
        }
        assert!(
            out[1].iter().any(|&b| b != 0),
            "m={m}: the copy's product is all zero, so nothing was multiplied"
        );
        if out[0] != out[1] {
            let diffs: Vec<(usize, u16, u16)> = out[0]
                .chunks(2)
                .zip(out[1].chunks(2))
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| {
                    (
                        i,
                        u16::from_le_bytes([a[0], a[1]]),
                        u16::from_le_bytes([b[0], b[1]]),
                    )
                })
                .collect();
            let rows: std::collections::BTreeSet<usize> =
                diffs.iter().map(|d| d.0 / N as usize).collect();
            panic!(
                "m={m}: the view over the wide bank reads differently from the copy: {} of {} differ, rows {:?}, first {:?}",
                diffs.len(),
                m * N,
                rows,
                &diffs[..diffs.len().min(6)]
            );
        }
        eprintln!("m={m}: view == copy over {} bf16 outputs", m * N);
    }
}

#[test]
fn the_leading_output_columns_land_in_place() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut rng = Lcg(0xface);
    let whole = planes(&mut rng, K_FULL, None);
    let mut keep = Vec::new();
    let full = bank(&device, &handles, &whole, K_FULL, K_FULL, &mut keep);
    // The leading `KEEP` rows of the bank, as the engine's `rows_of` view.
    let head = Bank {
        codes: Tensor {
            rows: KEEP,
            ..full.codes
        },
        scales: Tensor {
            rows: KEEP,
            ..full.scales
        },
        biases: full.biases.map(|b| Tensor { rows: KEEP, ..b }),
        ..full
    };
    let scratch_keep: RefCell<Vec<Buffer>> = RefCell::new(Vec::new());
    let plane = |rows: u32, width: u32, dtype: Dtype| -> Option<Tensor> {
        let bytes = u64::from(rows) * u64::from(width) * dtype.bits() / 8;
        let buffer = Buffer::zeroed(&device, bytes).ok()?;
        let handle = handles.bind(&buffer, 0, bytes).ok()?;
        scratch_keep.borrow_mut().push(buffer);
        Some(Tensor::new(handle, rows, width, dtype))
    };
    let precast = |rows, contraction| plane(rows, contraction, Dtype::F16);
    let partials = |rows, width| plane(rows, width, Dtype::F32);

    for m in [1u32, 4, 17, 64, 640] {
        let x: Vec<u8> = (0..m * K_FULL)
            .flat_map(|_| bf16(rng.unit()).to_le_bytes())
            .collect();
        let mut rows = Vec::new();
        let act = Tensor::new(
            bound(&device, &handles, &x, &mut rows),
            m,
            K_FULL,
            Dtype::Bf16,
        );
        let y_bytes = u64::from(m) * u64::from(N) * 2;
        let mut out = Vec::new();
        for (w, columns) in [(full, None), (head, Some(KEEP))] {
            let y_buffer = Buffer::zeroed(&device, whole_pages(y_bytes)).expect("y");
            let y = Tensor::new(
                handles.bind(&y_buffer, 0, y_bytes).expect("y"),
                m,
                N,
                Dtype::Bf16,
            );
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            let scratch = quant::Scratch {
                precast: &precast,
                partials: &partials,
            };
            match columns {
                None => quant::matmul(&sink, act, w, y, scratch, m.div_ceil(32) * 32),
                Some(c) => quant::matmul_columns(&sink, act, w, y, c, scratch, m.div_ceil(32) * 32),
            }
            .expect("the matmul launches");
            frame.commit().expect("the frame lands");
            out.push(handles.read(y.buf, y_bytes).expect("y read back"));
            scratch_keep.borrow_mut().clear();
            drop(y_buffer);
        }
        let (all, part) = (&out[0], &out[1]);
        for r in 0..m as usize {
            let span = r * N as usize * 2..(r + 1) * N as usize * 2;
            let (whole_row, part_row) = (&all[span.clone()], &part[span]);
            let cut = KEEP as usize * 2;
            assert!(
                whole_row[..cut] == part_row[..cut],
                "m={m} row {r}: the leading columns differ from the full product's"
            );
            assert!(
                part_row[cut..].iter().all(|&b| b == 0),
                "m={m} row {r}: a column past {KEEP} was written"
            );
        }
        eprintln!("m={m}: leading {KEEP} of {N} columns land in place, the rest untouched");
    }
}
