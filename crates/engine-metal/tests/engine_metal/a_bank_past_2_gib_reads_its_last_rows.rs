#![cfg(target_vendor = "apple")]

//! A quantized bank whose codes run past 2 GiB: its last rows lie past what
//! an `int` row offset reaches. Every quantized matmul path the dispatcher
//! takes over the row counts below (register and staged rows, one row, a
//! few rows folded, tiled)
//! lands those rows exactly as it lands the same planes laid as the bank's
//! first rows. And a bank whose row stride passes what a tile's `int`
//! offsets reach is refused by name.

use std::cell::RefCell;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::qmm;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

/// The contraction: 4-bit codes, 2 KiB a row.
const K: u32 = 4096;
const GROUP: u32 = 64;
/// The planes that move: `TAIL` rows of noise.
const TAIL: u32 = 64;
/// Rows of the bank: the tail's first row starts 2 GiB in.
const N: u32 = (1 << 31) / (K / 2) + TAIL;

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

struct Planes {
    codes: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

fn tail_planes(rng: &mut Lcg) -> Planes {
    let groups = (TAIL * (K / GROUP)) as usize;
    Planes {
        codes: (0..TAIL * K / 2).map(|_| rng.next() as u8).collect(),
        scales: (0..groups)
            .flat_map(|_| bf16(0.01 + rng.unit().abs() * 0.05))
            .collect(),
        biases: (0..groups).flat_map(|_| bf16(rng.unit() * 0.1)).collect(),
    }
}

#[test]
fn a_bank_past_2_gib_reads_its_last_rows() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut rng = Lcg(0x26b);
    let tail = tail_planes(&mut rng);

    let row = u64::from(K / 2);
    let group_row = u64::from(K / GROUP) * 2;
    let mut codes = Buffer::zeroed(&device, u64::from(N) * row).expect("the codes");
    let mut scales = Buffer::zeroed(&device, u64::from(N) * group_row).expect("the scales");
    let mut biases = Buffer::zeroed(&device, u64::from(N) * group_row).expect("the biases");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let bank = Bank {
        codes: Tensor::new(bind(&codes), N, K, Dtype::U4g64),
        mpp_codes: None,
        scales: Tensor::new(bind(&scales), N, K / GROUP, Dtype::Bf16),
        biases: Some(Tensor::new(bind(&biases), N, K / GROUP, Dtype::Bf16)),
        group: GROUP,
        bits: 4,
        ld: K,
    };
    // The tail's planes laid at bank row `first`, every other row zero.
    let mut lay = |first: u32, planes: Option<&Planes>| {
        let zeros = |bytes: usize| vec![0u8; bytes];
        let at = u64::from(first);
        let (c, s, b) = match planes {
            Some(p) => (p.codes.clone(), p.scales.clone(), p.biases.clone()),
            None => (
                zeros(tail.codes.len()),
                zeros(tail.scales.len()),
                zeros(tail.biases.len()),
            ),
        };
        codes.write(at * row, &c).expect("the codes land");
        scales.write(at * group_row, &s).expect("the scales land");
        biases.write(at * group_row, &b).expect("the biases land");
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

    let last = N - TAIL;
    // The dispatcher's own ladder (register and staged rows, tiled), then
    // the matrix-vector rung alone (one row; a few rows folded), then the
    // tiled rung over every row count.
    for ladder in [
        "",
        "[kernel]\nqmm = [\"qmv\"]",
        "[kernel]\nqmm = [\"tiled\", \"qmv\"]",
    ] {
        let tuning = engine_metal::chip::tuning(ladder, false);
        for m in [1u32, 4, 17, 64] {
            let x: Vec<u8> = (0..m * K).flat_map(|_| bf16(rng.unit())).collect();
            // The planes hold the rows a tile pads the batch to, as the
            // engine's activation does (`capacity_rows`).
            let capacity = m.div_ceil(32) * 32;
            let mut x_buffer =
                Buffer::zeroed(&device, u64::from(capacity) * u64::from(K) * 2).expect("x");
            x_buffer.write(0, &x).expect("x lands");
            let act = Tensor::new(bind(&x_buffer), m, K, Dtype::Bf16);
            let y_bytes = u64::from(m) * u64::from(N) * 2;
            let y_buffer =
                Buffer::zeroed(&device, u64::from(capacity) * u64::from(N) * 2).expect("y");
            let y = Tensor::new(bind(&y_buffer), m, N, Dtype::Bf16);
            // The tail's columns of each output row, with the tail at `first`.
            let run = |first: u32| -> Vec<u8> {
                let frame = device.frame().expect("a frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                qmm::matmul(
                    &sink,
                    &tuning,
                    act,
                    bank,
                    y,
                    qmm::Scratch {
                        precast: &precast,
                        partials: &partials,
                        table: None,
                    },
                    capacity,
                )
                .expect("the matmul launches");
                frame.commit().expect("the frame lands");
                scratch_keep.borrow_mut().clear();
                let out = handles.read(y.buf, y_bytes).expect("y read back");
                let width = u64::from(N) * 2;
                (0..u64::from(m))
                    .flat_map(|r| {
                        let at = (r * width + u64::from(first) * 2) as usize;
                        out[at..at + (TAIL * 2) as usize].to_vec()
                    })
                    .collect()
            };
            lay(last, Some(&tail));
            let far = run(last);
            lay(last, None);
            lay(0, Some(&tail));
            let near = run(0);
            lay(0, None);
            assert!(
                near.iter().any(|&b| b != 0),
                "{ladder:?} m={m}: the tail laid first multiplies to zero, so nothing was read"
            );
            let differ = far
                .chunks(2)
                .zip(near.chunks(2))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                differ,
                0,
                "{ladder:?} m={m}: the bank's last {TAIL} rows, {} bytes in, read differently from \
                 the same planes laid first ({differ} of {} outputs)",
                u64::from(last) * row,
                m * TAIL
            );
            eprintln!(
                "{ladder:?} m={m}: rows {} bytes into the bank land as the first rows",
                u64::from(last) * row
            );
        }
    }
}

#[test]
fn a_bank_row_stride_past_a_tiles_int_reach_is_refused() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    // 64 rows of 2^22 codes at up to 8 bytes a pack: 2^31 bytes, one past
    // an int. Nothing is read, so the planes are a page each.
    let ld = 1u32 << 22;
    let page = Buffer::zeroed(&device, 16384).expect("a page");
    let handle = handles.bind(&page, 0, 16384).expect("a handle");
    let bank = Bank {
        codes: Tensor::new(handle, 64, ld, Dtype::U4g64),
        mpp_codes: None,
        scales: Tensor::new(handle, 64, ld / GROUP, Dtype::Bf16),
        biases: Some(Tensor::new(handle, 64, ld / GROUP, Dtype::Bf16)),
        group: GROUP,
        bits: 4,
        ld,
    };
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(&device, &frame, &pipelines, &handles);
    let none = |_: u32, _: u32| -> Option<Tensor> { None };
    let refused = qmm::matmul(
        &sink,
        &engine_metal::chip::tuning("", false),
        Tensor::new(handle, 1, ld, Dtype::Bf16),
        bank,
        Tensor::new(handle, 1, 64, Dtype::Bf16),
        qmm::Scratch {
            precast: &none,
            partials: &none,
            table: None,
        },
        32,
    )
    .expect_err("the stride is past a tile's reach");
    let why = refused.to_string();
    assert!(
        why.contains("row stride is 4194304 codes") && why.contains("in-tile `int`"),
        "{why}"
    );
}
