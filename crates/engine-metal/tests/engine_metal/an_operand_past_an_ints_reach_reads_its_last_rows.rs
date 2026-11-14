#![cfg(target_vendor = "apple")]

//! Operands whose last rows lie past what an `int` offset reaches: a rotary
//! over more than 2^31 elements and a ternary (PTQ1_0) bank over more than
//! 2 GiB. Each turns or reads its last rows exactly as it does the same
//! values laid as its first rows.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::elemwise::rope;
use kernels_metal::linear::qmm;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

/// The rows at each end that carry values.
const ENDS: u32 = 8;

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

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    (((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16).to_le_bytes()
}

fn device() -> Option<(Context, Handles, Pipelines)> {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return None;
    };
    Some((device, Handles::new(), Pipelines::new()))
}

/// A rotary's tail over `ROWS` rows of 32 heads of 128: the last rows start
/// past 2^31 elements, and turn as the first rows do at the same positions.
#[test]
fn a_rotary_past_2_pow_31_elements_turns_its_last_rows() {
    const WIDTH: u32 = 32 * 128;
    const ROWS: u32 = (1 << 31) / WIDTH + ENDS;
    let Some((device, handles, pipelines)) = device() else {
        return;
    };
    let mut rng = Lcg(0x707e);
    let row = u64::from(WIDTH) * 2;
    let ends: Vec<u8> = (0..ENDS * WIDTH).flat_map(|_| bf16(rng.unit())).collect();
    let at: Vec<i32> = (0..ENDS).map(|r| 17 + 1009 * r as i32).collect();
    let last = ROWS - ENDS;

    let mut x = Buffer::zeroed(&device, u64::from(ROWS) * row).expect("the rows");
    x.write(0, &ends).expect("the first rows");
    x.write(u64::from(last) * row, &ends)
        .expect("the last rows");
    let mut positions = vec![0i32; ROWS as usize];
    positions[..ENDS as usize].copy_from_slice(&at);
    positions[last as usize..].copy_from_slice(&at);
    let position_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mut p = Buffer::zeroed(&device, position_bytes.len() as u64).expect("the positions");
    p.write(0, &position_bytes).expect("the positions land");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let xt = Tensor::new(bind(&x), ROWS, WIDTH, Dtype::Bf16);
    let pt = Tensor::new(bind(&p), ROWS, 1, Dtype::I32);
    let tail_rows = handles
        .bind(&x, u64::from(last) * row, u64::from(ENDS) * row)
        .expect("a handle on the last rows");

    for (rotary, yarn) in [
        (64, None),
        (
            128,
            Some(rope::Yarn {
                factor: 4.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 4096,
            }),
        ),
    ] {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        rope::partial_last(&sink, xt, pt, rotary, 128, 10_000.0, false, false, yarn)
            .expect("the rotary launches");
        frame.commit().expect("the frame lands");
        let bytes = (ENDS as usize) * row as usize;
        let first = handles.read(xt.buf, bytes as u64).expect("the first rows");
        let tail = handles
            .read(tail_rows, bytes as u64)
            .expect("the last rows");
        assert_ne!(&first[..], &ends[..], "the rotary turned nothing");
        let differ = first
            .chunks(2)
            .zip(tail.chunks(2))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differ,
            0,
            "rotary {rotary}: the last rows, {} elements in, turn otherwise than the first",
            u64::from(last) * u64::from(WIDTH)
        );
        eprintln!(
            "rotary {rotary}{}: rows {} elements in turn as the first rows",
            if yarn.is_some() { " (yarn)" } else { "" },
            u64::from(last) * u64::from(WIDTH)
        );
    }
}

/// A ternary bank of `K = 4096` (896 bytes a row) whose last rows start
/// past 2 GiB: they read as the same blocks laid as its first rows.
#[test]
fn a_ternary_bank_past_2_gib_reads_its_last_rows() {
    const K: u32 = 4096;
    const ROW: u64 = (K as u64 / 128) * 28;
    let tail_row = u32::try_from((1u64 << 31).div_ceil(ROW)).expect("rows");
    let n = tail_row + ENDS;
    let Some((device, handles, pipelines)) = device() else {
        return;
    };
    let mut rng = Lcg(0x7e57);
    // Blocks of random trits, each with a scale of 1/64 (fp16 0x2400).
    let blocks: Vec<u8> = (0..u64::from(ENDS) * ROW / 28)
        .flat_map(|_| {
            let mut block: Vec<u8> = (0..26).map(|_| rng.next() as u8).collect();
            block.extend_from_slice(&0x2400u16.to_le_bytes());
            block
        })
        .collect();
    let mut codes = Buffer::zeroed(&device, u64::from(n) * ROW).expect("the bank");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let hc = bind(&codes);
    let bank = Bank {
        codes: Tensor::new(hc, n, K, Dtype::Ptq1_0),
        scales: Tensor::new(hc, n, 1, Dtype::Bf16),
        biases: None,
        group: 128,
        bits: 2,
        mpp_codes: None,
        ld: K,
    };
    let none = |_: u32, _: u32| None;
    for m in [1u32, 4] {
        let xs: Vec<u8> = (0..m * K).flat_map(|_| bf16(rng.unit())).collect();
        let mut x = Buffer::zeroed(&device, xs.len() as u64).expect("x");
        x.write(0, &xs).expect("x lands");
        let y_bytes = u64::from(m) * u64::from(n) * 2;
        let y = Buffer::zeroed(&device, y_bytes).expect("y");
        let act = Tensor::new(bind(&x), m, K, Dtype::Bf16);
        let out = Tensor::new(bind(&y), m, n, Dtype::Bf16);
        let mut run = |first: u32| -> Vec<u8> {
            codes
                .write(u64::from(first) * ROW, &blocks)
                .expect("the blocks land");
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            qmm::matmul(
                &sink,
                &engine_metal::chip::tuning("", false),
                act,
                bank,
                out,
                qmm::Scratch {
                    precast: &none,
                    partials: &none,
                    table: None,
                },
                m,
            )
            .expect("the ternary matmul launches");
            frame.commit().expect("the frame lands");
            codes
                .write(u64::from(first) * ROW, &vec![0u8; blocks.len()])
                .expect("the blocks clear");
            let all = handles.read(out.buf, y_bytes).expect("y read back");
            (0..u64::from(m))
                .flat_map(|r| {
                    let at = ((r * u64::from(n) + u64::from(first)) * 2) as usize;
                    all[at..at + (ENDS * 2) as usize].to_vec()
                })
                .collect()
        };
        let far = run(tail_row);
        let near = run(0);
        assert!(
            near.iter().any(|&b| b != 0),
            "m={m}: the blocks laid first multiply to zero"
        );
        let differ = far
            .chunks(2)
            .zip(near.chunks(2))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differ,
            0,
            "m={m}: the bank's last rows, {} bytes in, read otherwise than its first",
            u64::from(tail_row) * ROW
        );
        eprintln!(
            "m={m}: ternary rows {} bytes in read as the first rows",
            u64::from(tail_row) * ROW
        );
    }
}
