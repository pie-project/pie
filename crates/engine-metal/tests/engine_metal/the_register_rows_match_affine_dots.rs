#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

#[test]
fn every_register_row_answers_an_affine_reference() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let tuning = engine_metal::chip::tuning("[kernel]\nqmm = [\"register\", \"qmv\"]", false);
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut checked = 0;
    let cases = [
        (256u32, 64u32),
        (6144, 5120),
        (512, 1024),
        (5120, 1280),
        (17408, 5120),
        (5120, 34816),
    ];
    for exact in [true, false] {
        for &(k, n) in &cases {
            let groups = k / 64;
            let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2).map(noise).collect();
            let stored = |value: f32| f32::from_bits(value.to_bits() & 0xffff0000);
            let scale = |at: u32| {
                if exact {
                    ((at % 7) + 1) as f32 / 128.0
                } else {
                    stored(0.0031 + f32::from(noise(u64::from(at))) * 0.000173)
                }
            };
            let bias = |at: u32| {
                if exact {
                    ((at % 9) as i32 - 4) as f32 / 64.0
                } else {
                    stored((f32::from(noise(u64::from(at) ^ 0x5678)) - 127.0) * 0.00157)
                }
            };
            let activation = |row: u32, at: u32| {
                if exact {
                    (((row * 17 + at * 3) % 29) as i32 - 14) as f32 / 16.0
                } else {
                    stored((f32::from(noise(u64::from(row * k + at) ^ 0xabcd)) - 127.0) * 0.0197)
                }
            };
            let scales: Vec<u8> = (0..n * groups).flat_map(|at| bf16(scale(at))).collect();
            let biases: Vec<u8> = (0..n * groups).flat_map(|at| bf16(bias(at))).collect();
            let upload = |bytes: &[u8]| {
                let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
                buffer.write(0, bytes).expect("upload");
                buffer
            };
            let codes_b = upload(&codes);
            let scales_b = upload(&scales);
            let biases_b = upload(&biases);
            let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
            let bank = Bank {
                mpp_codes: None,
                codes: Tensor::new(bind(&codes_b), n, k, Dtype::U4g64),
                scales: Tensor::new(bind(&scales_b), n, groups, Dtype::Bf16),
                biases: Some(Tensor::new(bind(&biases_b), n, groups, Dtype::Bf16)),
                group: 64,
                bits: 4,
                ld: k,
            };
            for m in [1u32, 5, 6, 8] {
                let cap = 8;
                let inputs: Vec<u8> = (0..cap)
                    .flat_map(|row| (0..k).flat_map(move |at| bf16(activation(row, at))))
                    .collect();
                let act_b = upload(&inputs);
                let output_bytes = u64::from(m) * u64::from(n) * 2;
                let out_b = upload(&vec![0xa5; output_bytes as usize + 64]);
                let (ha, ho) = (bind(&act_b), bind(&out_b));
                // The tile's input table: 8 x k bf16, k / 4 f32 of sums and the split's
                // arrival counters, left dirty for the table to zero.
                let table_rows = 16;
                let table_buffer = upload(&vec![0xa5; table_rows as usize * k as usize * 2]);
                let th = bind(&table_buffer);
                let precast = |rows: u32, width: u32| {
                    (rows <= table_rows && width == k)
                        .then(|| Tensor::new(th, rows, width, Dtype::F16))
                };
                let partial_bytes = u64::from(cap) * 8 * u64::from(n) * 4;
                let partial_buffer = upload(&vec![0xa5; partial_bytes as usize + 64]);
                let ph = bind(&partial_buffer);
                let partials = |rows, width| {
                    (rows <= cap * 8 && width == n)
                        .then(|| Tensor::new(ph, rows, width, Dtype::F32))
                };
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                quant::matmul(
                    &sink,
                    &tuning,
                    Tensor::new(ha, m, k, Dtype::Bf16),
                    bank,
                    Tensor::new(ho, m, n, Dtype::Bf16),
                    quant::Scratch {
                        precast: &precast,
                        partials: &partials,
                        table: None,
                    },
                    cap,
                )
                .expect("register");
                frame.commit().expect("commit");
                let got = handles.read(ho, output_bytes + 64).expect("read");
                assert!(
                    got[output_bytes as usize..].iter().all(|&b| b == 0xa5),
                    "write beyond output at M={m}, K={k}, N={n}"
                );
                for row in 0..m {
                    for col in (0..n).step_by(if n > 1024 { (n / 64) as usize } else { 1 }) {
                        let mut want = 0.0f64;
                        let mut absolute_sum = 0.0f64;
                        for at in 0..k {
                            let code_at = col * k + at;
                            let q = (codes[(code_at / 2) as usize] >> ((code_at % 2) * 4)) & 15;
                            let factor = col * groups + at / 64;
                            let product = f64::from(activation(row, at))
                                * (f64::from(q) * f64::from(scale(factor))
                                    + f64::from(bias(factor)));
                            want += product;
                            absolute_sum += product.abs();
                        }
                        let bits = (want as f32).to_bits();
                        let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                        let offset = ((row * n + col) * 2) as usize;
                        let actual = u16::from_le_bytes([got[offset], got[offset + 1]]);
                        let actual_f32 = f32::from_bits(u32::from(actual) << 16);
                        if exact {
                            assert_eq!(
                                actual, rounded,
                                "M={m} K={k} N={n} row={row} col={col}: {actual_f32} vs {want}"
                            );
                        } else {
                            let allowance = 0.00002 + 0.001 * absolute_sum / f64::from(k).sqrt();
                            assert!(
                                (f64::from(actual_f32) - want).abs()
                                    <= 0.004 * want.abs() + allowance,
                                "random M={m} K={k} N={n} row={row} col={col}: {actual_f32} vs {want}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!("checked {checked} register outputs against exact and random affine dots");
}

/// A 4-bit, 64-group bank of `n` rows of `k` codes, its planes seeded by `seed`.
fn bank_of(device: &Context, handles: &Handles, k: u32, n: u32, seed: u64) -> (Bank, [Buffer; 3]) {
    let groups = k / 64;
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2)
        .map(|at| noise(at ^ seed))
        .collect();
    let factor = |at: u32, salt: u64, base: f32, step: f32| {
        bf16(base + f32::from(noise(u64::from(at) ^ seed ^ salt)) * step)
    };
    let scales: Vec<u8> = (0..n * groups)
        .flat_map(|at| factor(at, 0x11, 0.0031, 0.000173))
        .collect();
    let biases: Vec<u8> = (0..n * groups)
        .flat_map(|at| factor(at, 0x5678, -0.2, 0.00157))
        .collect();
    let planes = [upload(&codes), upload(&scales), upload(&biases)];
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let bank = Bank {
        mpp_codes: None,
        codes: Tensor::new(bind(&planes[0]), n, k, Dtype::U4g64),
        scales: Tensor::new(bind(&planes[1]), n, groups, Dtype::Bf16),
        biases: Some(Tensor::new(bind(&planes[2]), n, groups, Dtype::Bf16)),
        group: 64,
        bits: 4,
        ld: k,
    };
    (bank, planes)
}

/// The matmuls of one activation share the register tile's input table: the
/// second reads the table the first laid (its split's arrival counters put
/// back to zero by the first's last partitions) and lands the very bits it
/// lands over a table of its own, whether it splits wider or narrower.
#[test]
fn a_table_one_matmul_laid_feeds_the_next() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let tuning = engine_metal::chip::tuning("[kernel]\nqmm = [\"register\", \"qmv\"]", false);
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    // (k, the first's columns, the second's): a layer's in-projection and its
    // gates, a query and a key, and the narrow one first.
    let cases = [
        (2048u32, 12288u32, 64u32),
        (2048, 8192, 512),
        (2048, 64, 12288),
        (4096, 2048, 1024),
    ];
    let mut checked = 0;
    for &(k, first_n, second_n) in &cases {
        let (first, _first_planes) = bank_of(&device, &handles, k, first_n, 0x1);
        let (second, _second_planes) = bank_of(&device, &handles, k, second_n, 0x2);
        for m in [1u32, 6, 8] {
            let cap = 8;
            let inputs: Vec<u8> = (0..cap * k)
                .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0xabcd)) - 127.0) * 0.0197))
                .collect();
            let act_b = upload(&inputs);
            let act = Tensor::new(bind(&act_b), m, k, Dtype::Bf16);
            let widest = first_n.max(second_n);
            let table_rows = 16 + (4 * widest / 64).div_ceil(2 * k);
            let table_b = upload(&vec![0xa5; (table_rows * k * 2) as usize]);
            let th = bind(&table_b);
            let partial_b = upload(&vec![0xa5; (cap * 8 * widest * 4) as usize]);
            let ph = bind(&partial_b);
            let precast = |rows: u32, width: u32| {
                (rows <= table_rows && width == k).then(|| Tensor::new(th, rows, width, Dtype::F16))
            };
            let partials = |rows: u32, width: u32| {
                (rows <= cap * 8 && width <= widest)
                    .then(|| Tensor::new(ph, rows, width, Dtype::F32))
            };
            // Laid by the first call, with the counters it zeroed; read by the second when
            // those cover its own.
            let laid = std::cell::Cell::new(None::<u32>);
            let kept = |rows: u32, width: u32, counters: u32| {
                let table = precast(rows, width)?;
                let standing = laid.get().is_some_and(|zeroed| zeroed >= counters);
                if !standing {
                    laid.set(Some(counters));
                }
                Some((table, standing))
            };
            let run =
                |bank: Bank,
                 n: u32,
                 out: u32,
                 table: Option<&dyn Fn(u32, u32, u32) -> Option<(Tensor, bool)>>| {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    quant::matmul(
                        &sink,
                        &tuning,
                        act,
                        bank,
                        Tensor::new(out, m, n, Dtype::Bf16),
                        quant::Scratch {
                            precast: &precast,
                            partials: &partials,
                            table,
                        },
                        cap,
                    )
                    .expect("register");
                    frame.commit().expect("commit");
                };
            let alone_b = upload(&vec![0x5a; (m * second_n * 2) as usize]);
            let first_b = upload(&vec![0x5a; (m * first_n * 2) as usize]);
            let shared_b = upload(&vec![0x5a; (m * second_n * 2) as usize]);
            let (alone, first_out, shared) = (bind(&alone_b), bind(&first_b), bind(&shared_b));
            run(second, second_n, alone, None);
            run(first, first_n, first_out, Some(&kept));
            run(second, second_n, shared, Some(&kept));
            let bytes = u64::from(m * second_n * 2);
            let want = handles.read(alone, bytes).expect("read");
            let got = handles.read(shared, bytes).expect("read");
            assert!(
                want == got,
                "k={k}, {first_n} then {second_n} columns, M={m}: the shared table lands other bits"
            );
            checked += 1;
        }
    }
    eprintln!("checked {checked} matmuls reading a table another laid");
}
