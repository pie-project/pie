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
            let run = |bank: Bank, n: u32, out: u32, table: Option<quant::KeptTable<'_>>| {
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

/// A residual add and RMS norm that lays the register tile's input table of
/// its normed rows lands the rows the plain norm lands, and a matmul reading
/// that table lands the bits it lands over the table it lays itself, the
/// counters of its split zeroed and the rows past the norm's left empty.
#[test]
fn a_norm_lays_the_table_its_projections_read() {
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
    let mut checked = 0;
    for &(k, n) in &[(2048u32, 12288u32), (2048, 64), (4096, 2048), (512, 2048)] {
        let (bank, _planes) = bank_of(&device, &handles, k, n, 0x3);
        for m in [1u32, 5, 8] {
            let rows = |salt: u64, scale: f32| -> Vec<u8> {
                (0..m * k)
                    .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ salt)) - 127.0) * scale))
                    .collect()
            };
            let x = rows(0x77, 0.031);
            let residual = rows(0x99, 0.017);
            let gains: Vec<u8> = (0..k)
                .flat_map(|at| bf16(0.5 + f32::from(noise(u64::from(at) ^ 0x42)) / 256.0))
                .collect();
            let (x_b, w_b) = (upload(&x), upload(&gains));
            let (plain_res_b, laid_res_b) = (upload(&residual), upload(&residual));
            let plain_out_b = upload(&vec![0; (m * k * 2) as usize]);
            let laid_out_b = upload(&vec![0; (m * k * 2) as usize]);
            let counters = n / 64;
            let table_rows = (17 * k + 4 * counters).div_ceil(2 * k);
            let table_b = upload(&vec![0xa5; (table_rows * k * 2) as usize]);
            let th = bind(&table_b);
            let tensor = |b: &Buffer| Tensor::new(bind(b), m, k, Dtype::Bf16);
            let weight = Tensor::new(bind(&w_b), 1, k, Dtype::Bf16);
            let (x_t, plain_out, laid_out) =
                (tensor(&x_b), tensor(&plain_out_b), tensor(&laid_out_b));
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            kernels_metal::elemwise::norm::residual_add_rmsnorm(
                &sink,
                x_t,
                tensor(&plain_res_b),
                weight,
                false,
                1e-6,
                plain_out,
            )
            .expect("norm");
            kernels_metal::elemwise::norm::residual_add_rmsnorm_table(
                &sink,
                x_t,
                tensor(&laid_res_b),
                weight,
                false,
                1e-6,
                laid_out,
                Tensor::new(th, table_rows, k, Dtype::F16),
                counters,
            )
            .expect("norm and table");
            frame.commit().expect("commit");
            let bytes = u64::from(m * k * 2);
            assert!(
                handles.read(plain_out.buf, bytes).expect("read")
                    == handles.read(laid_out.buf, bytes).expect("read"),
                "k={k} M={m}: the table-laying norm lands other rows"
            );
            let partial_b = upload(&vec![0xa5; (64 * n * 4) as usize]);
            let ph = bind(&partial_b);
            let own_b = upload(&vec![0xa5; (table_rows * k * 2) as usize]);
            let oh = bind(&own_b);
            let run = |out: u32, table: u32, laid: bool| {
                let precast = |rows: u32, width: u32| {
                    (rows <= table_rows && width == k)
                        .then(|| Tensor::new(table, rows, width, Dtype::F16))
                };
                let partials = |rows: u32, width: u32| {
                    (rows <= 64 && width == n).then(|| Tensor::new(ph, rows, width, Dtype::F32))
                };
                let kept = |rows: u32, width: u32, _: u32| Some((precast(rows, width)?, laid));
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                quant::matmul(
                    &sink,
                    &tuning,
                    plain_out,
                    bank,
                    Tensor::new(out, m, n, Dtype::Bf16),
                    quant::Scratch {
                        precast: &precast,
                        partials: &partials,
                        table: Some(&kept),
                    },
                    8,
                )
                .expect("register");
                frame.commit().expect("commit");
            };
            let alone_b = upload(&vec![0x5a; (m * n * 2) as usize]);
            let read_b = upload(&vec![0x5a; (m * n * 2) as usize]);
            run(bind(&alone_b), oh, false);
            run(bind(&read_b), th, true);
            let bytes = u64::from(m * n * 2);
            assert!(
                handles.read(bind(&alone_b), bytes).expect("read")
                    == handles.read(bind(&read_b), bytes).expect("read"),
                "k={k} N={n} M={m}: the matmul over the norm's table lands other bits"
            );
            // The inputs and their sums, byte for byte the table the matmul laid itself;
            // then the counters, zeroed.
            let laid = handles
                .read(th, u64::from(table_rows * k * 2))
                .expect("read");
            let own = handles.read(oh, u64::from(17 * k)).expect("read");
            assert!(
                laid[..(17 * k) as usize] == own[..],
                "k={k} M={m}: the norm's table is not the one the matmul lays"
            );
            assert!(
                laid[(17 * k) as usize..(17 * k + 4 * counters) as usize]
                    .iter()
                    .all(|&b| b == 0),
                "k={k} M={m}: the norm left an arrival counter standing"
            );
            checked += 1;
        }
    }
    eprintln!("checked {checked} matmuls over a table their norm laid");
}

/// A swiglu that lays the register tile's input table of its rows lands the
/// rows the plain swiglu lands and the table the down projection lays over
/// them, byte for byte, its split's counters zeroed.
#[test]
fn a_swiglu_lays_the_table_its_down_reads() {
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
    let mut checked = 0;
    for &(k, n) in &[(512u32, 2048u32), (1024, 4096), (6144, 2048)] {
        let (bank, _planes) = bank_of(&device, &handles, k, n, 0x4);
        for m in [1u32, 5, 8] {
            let packed: Vec<u8> = (0..m * 2 * k)
                .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0x31)) - 127.0) * 0.043))
                .collect();
            let packed_t = Tensor::new(bind(&upload(&packed)), m, 2 * k, Dtype::Bf16);
            let plain = Tensor::new(
                bind(&upload(&vec![0; (m * k * 2) as usize])),
                m,
                k,
                Dtype::Bf16,
            );
            let laid = Tensor::new(
                bind(&upload(&vec![0; (m * k * 2) as usize])),
                m,
                k,
                Dtype::Bf16,
            );
            let counters = n / 64;
            let table_rows = (17 * k + 4 * counters).div_ceil(2 * k);
            let th = bind(&upload(&vec![0xa5; (table_rows * k * 2) as usize]));
            let oh = bind(&upload(&vec![0xa5; (table_rows * k * 2) as usize]));
            let ph = bind(&upload(&vec![0xa5; (64 * n * 4) as usize]));
            let yh = bind(&upload(&vec![0x5a; (m * n * 2) as usize]));
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            kernels_metal::linear::mlp::swiglu(&sink, packed_t, k, plain).expect("swiglu");
            kernels_metal::linear::mlp::swiglu_table(
                &sink,
                packed_t,
                k,
                laid,
                Tensor::new(th, table_rows, k, Dtype::F16),
                counters,
            )
            .expect("swiglu and table");
            let precast = |rows: u32, width: u32| {
                (rows <= table_rows && width == k).then(|| Tensor::new(oh, rows, width, Dtype::F16))
            };
            let partials = |rows: u32, width: u32| {
                (rows <= 64 && width == n).then(|| Tensor::new(ph, rows, width, Dtype::F32))
            };
            quant::matmul(
                &sink,
                &tuning,
                plain,
                bank,
                Tensor::new(yh, m, n, Dtype::Bf16),
                quant::Scratch {
                    precast: &precast,
                    partials: &partials,
                    table: None,
                },
                8,
            )
            .expect("register");
            frame.commit().expect("commit");
            let bytes = u64::from(m * k * 2);
            assert!(
                handles.read(plain.buf, bytes).expect("read")
                    == handles.read(laid.buf, bytes).expect("read"),
                "k={k} M={m}: the table-laying swiglu lands other rows"
            );
            let table = handles
                .read(th, u64::from(table_rows * k * 2))
                .expect("read");
            let own = handles.read(oh, u64::from(17 * k)).expect("read");
            assert!(
                table[..(17 * k) as usize] == own[..],
                "k={k} M={m}: the swiglu's table is not the one the matmul lays"
            );
            assert!(
                table[(17 * k) as usize..(17 * k + 4 * counters) as usize]
                    .iter()
                    .all(|&b| b == 0),
                "k={k} M={m}: the swiglu left an arrival counter standing"
            );
            checked += 1;
        }
    }
    eprintln!("checked {checked} tables a swiglu laid");
}

/// The table a matmul of `act` (`m` x `k`) over an `n`-column bank lays, byte
/// for byte as far as its inputs and sums go.
fn own_table(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    act: Tensor,
    n: u32,
) -> Vec<u8> {
    let (m, k) = (act.rows, act.width);
    let tuning = engine_metal::chip::tuning("[kernel]\nqmm = [\"register\", \"qmv\"]", false);
    let (bank, _planes) = bank_of(device, handles, k, n, 0x5);
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let table_rows = (17 * k + 4 * (n / 64)).div_ceil(2 * k);
    let (table_b, partial_b, out_b) = (
        upload(&vec![0xa5; (table_rows * k * 2) as usize]),
        upload(&vec![0xa5; (64 * n * 4) as usize]),
        upload(&vec![0; (m * n * 2) as usize]),
    );
    let (th, ph) = (bind(&table_b), bind(&partial_b));
    let precast = |rows: u32, width: u32| {
        (rows <= table_rows && width == k).then(|| Tensor::new(th, rows, width, Dtype::F16))
    };
    let partials = |rows: u32, width: u32| {
        (rows <= 64 && width == n).then(|| Tensor::new(ph, rows, width, Dtype::F32))
    };
    let frame = device.frame().expect("frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    quant::matmul(
        &sink,
        &tuning,
        act,
        bank,
        Tensor::new(bind(&out_b), m, n, Dtype::Bf16),
        quant::Scratch {
            precast: &precast,
            partials: &partials,
            table: None,
        },
        8,
    )
    .expect("register");
    frame.commit().expect("commit");
    handles.read(th, u64::from(17 * k)).expect("read")
}

/// The attention gate and the gated RMS norm that lay the register tile's
/// input table of their rows land the rows the plain ops land and the table
/// the out projection lays over them, byte for byte, the counters zeroed.
#[test]
fn a_gate_and_a_gated_norm_lay_the_table_their_out_projection_reads() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let n = 2048u32;
    let counters = n / 64;
    let mut checked = 0;
    for k in [4096u32, 1024] {
        let table_rows = (17 * k + 4 * counters).div_ceil(2 * k);
        for m in [1u32, 5, 8] {
            let plane = |salt: u64, scale: f32| -> Vec<u8> {
                (0..m * k)
                    .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ salt)) - 127.0) * scale))
                    .collect()
            };
            let check = |plain: Tensor, laid: Tensor, th: u32, what: &str| {
                let bytes = u64::from(m * k * 2);
                assert!(
                    handles.read(plain.buf, bytes).expect("read")
                        == handles.read(laid.buf, bytes).expect("read"),
                    "{what} k={k} M={m}: the table-laying op lands other rows"
                );
                let table = handles
                    .read(th, u64::from(table_rows * k * 2))
                    .expect("read");
                let own = own_table(&device, &handles, &pipelines, plain, n);
                let first = table[..(17 * k) as usize]
                    .iter()
                    .zip(&own)
                    .position(|(a, b)| a != b);
                assert!(
                    first.is_none(),
                    "{what} k={k} M={m}: another table from byte {first:?}"
                );
                assert!(
                    table[(17 * k) as usize..(17 * k + 4 * counters) as usize]
                        .iter()
                        .all(|&b| b == 0),
                    "{what} k={k} M={m}: an arrival counter left standing"
                );
            };
            // The attention gate, in place.
            let attn = plane(0x61, 0.05);
            let gate = Tensor::new(bind(&upload(&plane(0x62, 0.09))), m, k, Dtype::Bf16);
            let plain = Tensor::new(bind(&upload(&attn)), m, k, Dtype::Bf16);
            let laid = Tensor::new(bind(&upload(&attn)), m, k, Dtype::Bf16);
            let th = bind(&upload(&vec![0xa5; (table_rows * k * 2) as usize]));
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            kernels_metal::elemwise::gate::sigmoid_mul(&sink, gate, plain).expect("gate");
            kernels_metal::elemwise::gate::sigmoid_mul_table(
                &sink,
                gate,
                laid,
                Tensor::new(th, table_rows, k, Dtype::F16),
                counters,
            )
            .expect("gate and table");
            frame.commit().expect("commit");
            check(plain, laid, th, "gate");
            // The gated norm over f32 rows of 128-wide heads.
            let x: Vec<u8> = (0..m * k)
                .flat_map(|at| {
                    ((f32::from(noise(u64::from(at) ^ 0x71)) - 127.0) * 0.013).to_le_bytes()
                })
                .collect();
            let w: Vec<u8> = (0..128u32)
                .flat_map(|at| (0.6 + f32::from(noise(u64::from(at) ^ 0x72)) / 300.0).to_le_bytes())
                .collect();
            let x_t = Tensor::new(bind(&upload(&x)), m, k, Dtype::F32);
            let w_t = Tensor::new(bind(&upload(&w)), 1, 128, Dtype::F32);
            let z = Tensor::new(bind(&upload(&plane(0x73, 0.07))), m, k, Dtype::Bf16);
            let plain = Tensor::new(
                bind(&upload(&vec![0; (m * k * 2) as usize])),
                m,
                k,
                Dtype::Bf16,
            );
            let laid = Tensor::new(
                bind(&upload(&vec![0; (m * k * 2) as usize])),
                m,
                k,
                Dtype::Bf16,
            );
            let th = bind(&upload(&vec![0xa5; (table_rows * k * 2) as usize]));
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            kernels_metal::elemwise::norm::rmsnorm_gated(
                &sink, x_t, z, w_t, 128, 1e-6, false, plain,
            )
            .expect("gated norm");
            kernels_metal::elemwise::norm::rmsnorm_gated_table(
                &sink,
                x_t,
                z,
                w_t,
                128,
                1e-6,
                laid,
                Tensor::new(th, table_rows, k, Dtype::F16),
                counters,
            )
            .expect("gated norm and table");
            frame.commit().expect("commit");
            check(plain, laid, th, "gated norm");
            checked += 2;
        }
    }
    eprintln!("checked {checked} tables a gate or a gated norm laid");
}

/// A plain RMS norm (scaled by its weight or by one plus it) that lays the
/// register tile's input table lands the rows the plain norm lands and the table its projections lay, the counters zeroed.
#[test]
fn a_plain_norm_lays_the_table_its_projections_read() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let n = 4096u32;
    let counters = n / 64;
    for k in [2048u32, 512] {
        let table_rows = (17 * k + 4 * counters).div_ceil(2 * k);
        for (m, plus_one) in [(1u32, false), (6, true), (8, false), (8, true)] {
            let x: Vec<u8> = (0..m * k)
                .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0x81)) - 127.0) * 0.37))
                .collect();
            let w: Vec<u8> = (0..k)
                .flat_map(|at| bf16(0.3 + f32::from(noise(u64::from(at) ^ 0x82)) / 97.0))
                .collect();
            let x_t = Tensor::new(bind(&upload(&x)), m, k, Dtype::Bf16);
            let w_t = Tensor::new(bind(&upload(&w)), 1, k, Dtype::Bf16);
            let plain = Tensor::new(
                bind(&upload(&vec![0; (m * k * 2) as usize])),
                m,
                k,
                Dtype::Bf16,
            );
            let laid = Tensor::new(
                bind(&upload(&vec![0; (m * k * 2) as usize])),
                m,
                k,
                Dtype::Bf16,
            );
            let th = bind(&upload(&vec![0xa5; (table_rows * k * 2) as usize]));
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            if plus_one {
                kernels_metal::elemwise::norm::rmsnorm_plus_one(&sink, x_t, w_t, 1e-6, plain)
            } else {
                kernels_metal::elemwise::norm::rmsnorm(&sink, x_t, w_t, 1e-6, plain)
            }
            .expect("norm");
            kernels_metal::elemwise::norm::rmsnorm_table(
                &sink,
                x_t,
                w_t,
                plus_one,
                1e-6,
                laid,
                Tensor::new(th, table_rows, k, Dtype::F16),
                counters,
            )
            .expect("norm and table");
            frame.commit().expect("commit");
            let bytes = u64::from(m * k * 2);
            assert!(
                handles.read(plain.buf, bytes).expect("read")
                    == handles.read(laid.buf, bytes).expect("read"),
                "k={k} M={m}: the table-laying norm lands other rows"
            );
            let table = handles
                .read(th, u64::from(table_rows * k * 2))
                .expect("read");
            let own = own_table(&device, &handles, &pipelines, plain, n);
            assert!(
                table[..(17 * k) as usize] == own[..],
                "k={k} M={m}: another table"
            );
            assert!(
                table[(17 * k) as usize..(17 * k + 4 * counters) as usize]
                    .iter()
                    .all(|&b| b == 0),
                "k={k} M={m}: an arrival counter left standing"
            );
        }
    }
}

/// Every table producer's check on the GPU at hand, run by name with
/// `--ignored` (as the microbenchmarks are) to read them on another chip.
#[test]
#[ignore = "the table producers' checks, run on demand on a named GPU"]
fn register_tables_on_this_gpu() {
    a_table_one_matmul_laid_feeds_the_next();
    a_norm_lays_the_table_its_projections_read();
    a_plain_norm_lays_the_table_its_projections_read();
    a_swiglu_lays_the_table_its_down_reads();
    a_gate_and_a_gated_norm_lay_the_table_their_out_projection_reads();
    eprintln!("every register table producer matched qmm_sg_table on this GPU");
}

/// A segment launch lands each of its matmuls' bits as that matmul's own
/// launch lands them: a GDN in-projection and its gates, a query, key and
/// value, and segments of other contractions, at five to eight rows, each at
/// its own split (one, two, four or eight partitions side by side), over one
/// table; a projection's two halves as two segments of its bank land its
/// columns; and every walk of the tile lands the default walk's bits.
#[test]
fn a_segment_launch_lands_each_matmuls_bits() {
    use kernels_metal::linear::quant::Segment;
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len().max(16) as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let partials_b = upload(&vec![0xa5; 8 << 20]);
    let partials = Tensor::new(bind(&partials_b), 1, 2 << 20, Dtype::F32);
    let mut checked = 0;
    for (k, widths) in [
        (2048u32, &[12288u32, 64][..]),
        (2048, &[8192, 512, 512][..]),
        (4096, &[2048, 64, 1024][..]),
        (512, &[1024, 2048][..]),
    ] {
        let banks: Vec<(Bank, [Buffer; 3])> = widths
            .iter()
            .enumerate()
            .map(|(i, &n)| bank_of(&device, &handles, k, n, 0x51 + i as u64))
            .collect();
        for m in [5u32, 8] {
            let x: Vec<u8> = (0..m * k)
                .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0x2468)) - 127.0) * 0.021))
                .collect();
            let x_b = upload(&x);
            let act = Tensor::new(bind(&x_b), m, k, Dtype::Bf16);
            let outputs = |fill: u8| -> Vec<Buffer> {
                widths
                    .iter()
                    .map(|&n| upload(&vec![fill; (m * n * 2) as usize]))
                    .collect()
            };
            let segments = |ys: &[Buffer]| -> Vec<Segment> {
                banks
                    .iter()
                    .zip(ys)
                    .zip(widths)
                    .map(|(((bank, _), y), &n)| Segment {
                        w: *bank,
                        first: 0,
                        y: Tensor::new(bind(y), m, n, Dtype::Bf16),
                        split: quant::register_split(n as i32, k as i32, Some(20)),
                    })
                    .collect()
            };
            let (counters, floats) = quant::segments_scratch(&segments(&outputs(0)), m);
            assert!(
                floats <= 2 << 20,
                "the test's partials hold the segments' sums"
            );
            let rows = quant::register_table_rows(k as i32, counters as i32);
            let table_b = upload(&vec![0xa5; (rows * k * 2) as usize]);
            let table = Tensor::new(bind(&table_b), rows, k, Dtype::Bf16);
            let read = |ys: &[Buffer]| -> Vec<Vec<u8>> {
                ys.iter()
                    .map(|y| handles.read(bind(y), y.bytes()).expect("read"))
                    .collect()
            };
            // each matmul its own launch, on the default walk: the reference
            let alone = outputs(0x5a);
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            quant::register_table(&sink, act, table, m as i32, k as i32, counters as i32)
                .expect("table");
            for s in segments(&alone) {
                let n = s.y.width as i32;
                quant::register_matmul(
                    &sink,
                    s.w,
                    s.w.biases.expect("biases"),
                    table,
                    s.y,
                    (m as i32, n, k as i32, n, k as i32),
                    s.split,
                    (s.split > 1).then_some(partials),
                    quant::REGISTER_WALK,
                )
                .expect("matmul");
            }
            frame.commit().expect("commit");
            let reference = read(&alone);
            for walk in [quant::REGISTER_WALK, (2, 2), (4, 4), (4, 2)] {
                let (own, one) = (outputs(0x5a), outputs(0x3c));
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for s in segments(&own) {
                    let n = s.y.width as i32;
                    quant::register_matmul(
                        &sink,
                        s.w,
                        s.w.biases.expect("biases"),
                        table,
                        s.y,
                        (m as i32, n, k as i32, n, k as i32),
                        s.split,
                        (s.split > 1).then_some(partials),
                        walk,
                    )
                    .expect("matmul");
                }
                quant::register_segments(
                    &sink,
                    &segments(&one),
                    table,
                    m as i32,
                    k as i32,
                    Some(partials),
                    walk,
                )
                .expect("segments");
                frame.commit().expect("commit");
                assert!(
                    read(&own) == reference,
                    "K={k} {widths:?} M={m}: walk {walk:?} lands other bits than the default walk"
                );
                assert!(
                    read(&one) == reference,
                    "K={k} {widths:?} M={m}: the segment launch on walk {walk:?} lands other bits \
                     than its matmuls' own launches"
                );
                checked += 1;
            }
            // the launches put every arrival counter back to zero
            let laid = handles.read(bind(&table_b), table_b.bytes()).expect("read");
            let at = (17 * k) as usize;
            assert!(
                laid[at..at + 4 * counters as usize].iter().all(|&b| b == 0),
                "K={k} {widths:?} M={m}: a segment launch left an arrival counter standing"
            );
        }
    }
    // A projection's rows split in two halves as two segments of its bank at the projection's
    // own split, beside its gates: each half the whole projection's columns, to the bit.
    let (k, n, cut) = (2048u32, 12288u32, 8192u32);
    let (whole, _whole_planes) = bank_of(&device, &handles, k, n, 0x77);
    let (gates, _gates_planes) = bank_of(&device, &handles, k, 64, 0x78);
    for m in [5u32, 8] {
        let x: Vec<u8> = (0..m * k)
            .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0x1357)) - 127.0) * 0.021))
            .collect();
        let x_b = upload(&x);
        let act = Tensor::new(bind(&x_b), m, k, Dtype::Bf16);
        let split = quant::register_split(n as i32, k as i32, Some(20));
        let out = |width: u32| {
            let b = upload(&vec![0x3c; (m * width * 2) as usize]);
            let t = Tensor::new(bind(&b), m, width, Dtype::Bf16);
            (b, t)
        };
        let ((whole_b, whole_y), (gate_b, gate_y)) = (out(n), out(64));
        let ((left_b, left_y), (right_b, right_y), (gate2_b, gate2_y)) =
            (out(cut), out(n - cut), out(64));
        let segments = [
            Segment {
                w: whole,
                first: 0,
                y: left_y,
                split,
            },
            Segment {
                w: whole,
                first: cut,
                y: right_y,
                split,
            },
            Segment {
                w: gates,
                first: 0,
                y: gate2_y,
                split: quant::register_split(64, k as i32, Some(20)),
            },
        ];
        let (counters, _) = quant::segments_scratch(&segments, m);
        let rows = quant::register_table_rows(k as i32, counters as i32);
        let table_b = upload(&vec![0xa5; (rows * k * 2) as usize]);
        let table = Tensor::new(bind(&table_b), rows, k, Dtype::Bf16);
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        quant::register_table(&sink, act, table, m as i32, k as i32, counters as i32)
            .expect("table");
        for (w, y, split) in [(whole, whole_y, split), (gates, gate_y, segments[2].split)] {
            let width = y.width as i32;
            quant::register_matmul(
                &sink,
                w,
                w.biases.expect("biases"),
                table,
                y,
                (m as i32, width, k as i32, width, k as i32),
                split,
                (split > 1).then_some(partials),
                quant::REGISTER_WALK,
            )
            .expect("matmul");
        }
        quant::register_segments(
            &sink,
            &segments,
            table,
            m as i32,
            k as i32,
            Some(partials),
            quant::REGISTER_WALK,
        )
        .expect("segments");
        frame.commit().expect("commit");
        let read = |b: &Buffer| handles.read(bind(b), b.bytes()).expect("read");
        let (whole_bits, left_bits, right_bits) = (read(&whole_b), read(&left_b), read(&right_b));
        for r in 0..m as usize {
            let row = &whole_bits[r * n as usize * 2..(r + 1) * n as usize * 2];
            assert!(
                row[..cut as usize * 2]
                    == left_bits[r * cut as usize * 2..(r + 1) * cut as usize * 2]
                    && row[cut as usize * 2..]
                        == right_bits[r * (n - cut) as usize * 2..(r + 1) * (n - cut) as usize * 2],
                "M={m} row {r}: a half lands other bits than the whole projection's columns"
            );
        }
        assert!(
            read(&gate_b) == read(&gate2_b),
            "M={m}: the gates beside the halves land other bits"
        );
        checked += 1;
    }
    eprintln!("checked {checked} segment launches against their matmuls' own");
}
