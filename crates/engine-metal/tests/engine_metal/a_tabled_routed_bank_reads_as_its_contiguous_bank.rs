#![cfg(target_vendor = "apple")]
//! A quantized routed bank read through the weight cache's expert table
//! (`kernels_metal::linear::moe::BankTable`: each plane's experts, a piece
//! at a time, scattered over a pool in reverse order) lands the bits its
//! contiguous bank lands, on every routed arm: the per-route qmv (affine
//! 64/{2,4,8}, 32/2, 128/2 and MXFP4, with and without the expert bias), the
//! batched matmul (the fp16 and stamped tiled affine kernels, the affine and
//! MXFP4 MPP kernels, the tiled MXFP4 kernel with and without its bias) and
//! the register tiles (gate-up and down); one piece an expert and two. The
//! contiguous bank is scribbled over before the tabled run, so a read that
//! did not go through the table shows. The use record counts each route.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::moe::{self, BankTable, RoutedScratch};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scheme {
    Affine { bits: u32, group: u32 },
    Mxfp4,
}

impl Scheme {
    fn codes_dtype(self) -> Dtype {
        match self {
            Scheme::Affine { bits: 8, .. } => Dtype::U8g64,
            Scheme::Affine { bits: 2, group: 32 } => Dtype::U2g32,
            Scheme::Affine { bits: 2, group: 128 } => Dtype::U2g128,
            Scheme::Affine { bits: 2, .. } => Dtype::U2g64,
            Scheme::Affine { .. } | Scheme::Mxfp4 => Dtype::U4g64,
        }
    }
    fn bits(self) -> u32 {
        match self {
            Scheme::Affine { bits, .. } => bits,
            Scheme::Mxfp4 => 4,
        }
    }
    fn group(self) -> u32 {
        match self {
            Scheme::Affine { group, .. } => group,
            Scheme::Mxfp4 => 32,
        }
    }
}

/// One bank's planes, each `experts * n` rows: (bytes, row bytes).
struct Planes {
    planes: Vec<(Vec<u8>, u64)>,
}

/// The planes of a bank of `experts` experts of `n` rows of `k` inputs:
/// codes, scales, the affine zero points, and the expert bias where
/// `biased`.
fn planes(scheme: Scheme, experts: u32, n: u32, k: u32, biased: bool, seed: u64) -> Planes {
    let rows = u64::from(experts) * u64::from(n);
    let code_row = u64::from(k) * u64::from(scheme.bits()) / 8;
    let groups = u64::from(k / scheme.group());
    let codes: Vec<u8> = (0..rows * code_row).map(|at| noise(at ^ seed)).collect();
    let mut out = vec![(codes, code_row)];
    match scheme {
        Scheme::Affine { .. } => {
            let scales = (0..rows * groups)
                .flat_map(|at| bf16(0.002 + f32::from(noise(at ^ seed ^ 0x51)) * 0.0001))
                .collect();
            let zeros = (0..rows * groups)
                .flat_map(|at| bf16((f32::from(noise(at ^ seed ^ 0x77)) - 127.0) * 0.001))
                .collect();
            out.push((scales, groups * 2));
            out.push((zeros, groups * 2));
        }
        Scheme::Mxfp4 => {
            let exponents = (0..rows * groups)
                .map(|at| 127 - 4 + noise(at ^ seed ^ 0x33) % 5)
                .collect();
            out.push((exponents, groups));
        }
    }
    if biased {
        let bias = (0..rows)
            .flat_map(|at| bf16((f32::from(noise(at ^ seed ^ 0x99)) - 127.0) * 0.01))
            .collect();
        out.push((bias, 2));
    }
    Planes { planes: out }
}

/// The store a case reads: each plane contiguous, then every plane's
/// experts again, a piece at a time, in reverse order; and each plane's
/// address table over the scattered copy.
struct Store {
    buffer: Buffer,
    /// Each plane's contiguous offset and bytes.
    whole: Vec<(u64, u64)>,
    tables: Vec<Buffer>,
}

fn store(device: &Context, planes: &Planes, experts: u32, n: u32, pieces: u32) -> Store {
    let align = |at: u64| at.next_multiple_of(256);
    let mut at = 0u64;
    let mut whole = Vec::new();
    for (bytes, _) in &planes.planes {
        whole.push((at, bytes.len() as u64));
        at = align(at + bytes.len() as u64);
    }
    let piece_rows = n / pieces;
    // Entries (plane, expert, piece) from the last on, each at its own offset.
    let mut placed = Vec::new();
    for (p, (_, row)) in planes.planes.iter().enumerate() {
        for e in (0..experts).rev() {
            for j in (0..pieces).rev() {
                placed.push((p, e, j, at));
                at = align(at + u64::from(piece_rows) * row);
            }
        }
    }
    let mut buffer = Buffer::zeroed(device, at).expect("the store");
    for ((bytes, _), (offset, _)) in planes.planes.iter().zip(&whole) {
        buffer.write(*offset, bytes).expect("a plane");
    }
    let mut tables: Vec<Vec<u64>> = planes
        .planes
        .iter()
        .map(|_| vec![0; (experts * pieces) as usize])
        .collect();
    for &(p, e, j, offset) in &placed {
        let (bytes, row) = &planes.planes[p];
        let from = (u64::from(e) * u64::from(n) + u64::from(j * piece_rows)) * row;
        let len = u64::from(piece_rows) * row;
        buffer
            .write(offset, &bytes[from as usize..(from + len) as usize])
            .expect("a piece");
        tables[p][(e * pieces + j) as usize] = buffer.address_at(offset).expect("an address");
    }
    let tables = tables
        .iter()
        .map(|entries| {
            let bytes: Vec<u8> = entries.iter().flat_map(|a| a.to_le_bytes()).collect();
            let mut table = Buffer::zeroed(device, bytes.len() as u64).expect("a table");
            table.write(0, &bytes).expect("the table");
            table
        })
        .collect();
    Store {
        buffer,
        whole,
        tables,
    }
}

/// The bank over `store`'s contiguous planes, and its table.
#[allow(clippy::too_many_arguments)]
fn bank_of(
    handles: &Handles,
    store: &Store,
    scheme: Scheme,
    experts: u32,
    n: u32,
    k: u32,
    pieces: u32,
    hits: Option<Tensor>,
) -> (Bank, Option<Tensor>, BankTable) {
    let rows = experts * n;
    let bind = |p: usize, width: u32, dtype: Dtype| {
        let (offset, bytes) = store.whole[p];
        Tensor::new(
            handles.bind(&store.buffer, offset, bytes).expect("bind"),
            rows,
            width,
            dtype,
        )
    };
    let groups = k / scheme.group();
    let affine = matches!(scheme, Scheme::Affine { .. });
    let bank = Bank {
        mpp_codes: None,
        codes: bind(0, k, scheme.codes_dtype()),
        scales: bind(1, groups, if affine { Dtype::Bf16 } else { Dtype::E8m0 }),
        biases: affine.then(|| bind(2, groups, Dtype::Bf16)),
        group: scheme.group(),
        bits: scheme.bits(),
        ld: k,
    };
    let bias_plane = if affine { 3 } else { 2 };
    let bias = (store.whole.len() > bias_plane).then(|| {
        let (offset, bytes) = store.whole[bias_plane];
        Tensor::new(
            handles.bind(&store.buffer, offset, bytes).expect("bind"),
            experts,
            n,
            Dtype::Bf16,
        )
    });
    let table = |p: usize| {
        let buffer = &store.tables[p];
        Tensor::new(
            handles.bind(buffer, 0, buffer.bytes()).expect("bind"),
            experts * pieces,
            1,
            Dtype::I64,
        )
    };
    let tabled = BankTable {
        codes: table(0),
        scales: table(1),
        biases: affine.then(|| table(2)),
        bias: bias.map(|_| table(bias_plane)),
        pieces,
        piece_rows: n / pieces,
        hits,
    };
    (bank, bias, tabled)
}

/// Scribbles over `store`'s contiguous planes.
fn scribble(store: &mut Store) {
    for &(offset, bytes) in &store.whole {
        store
            .buffer
            .write(offset, &vec![0x5a; bytes as usize])
            .expect("scribble");
    }
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    Qmv,
    Batched(&'static str),
}

fn tuning_of(arm: Arm) -> kernels_metal::Tuning {
    let text = match arm {
        Arm::Batched("mpp") => {
            r#"
[kernel]
moe.path = "mpp"
moe.tile_rows = [{ when = "per_expert >= 32", value = 32 }, { value = 16 }]
"#
        }
        Arm::Batched("stamped") => {
            r#"
[kernel]
qmm_fp16 = false
moe.tile_rows = [{ when = "per_expert >= 32", value = 32 }, { value = 16 }]
"#
        }
        _ => {
            r#"
[kernel]
moe.tile_rows = [{ when = "per_expert >= 32", value = 32 }, { value = 16 }]
"#
        }
    };
    engine_metal::chip::tuning(text, false)
}

/// The rows a case lands, read back.
fn run(scheme: Scheme, arm: Arm, biased: bool, pieces: u32) -> (Vec<u8>, Vec<u8>) {
    let device = Context::bind().expect("the system device");
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let tuning = tuning_of(arm);
    let (experts, n, k) = (4u32, 256u32, 256u32);
    let (tokens, top_k) = match arm {
        Arm::Qmv => (3u32, 2u32),
        Arm::Batched(_) => (80, 2),
    };
    let pairs = tokens * top_k;
    let planes = planes(scheme, experts, n, k, biased, 0xbeef);
    let mut store = store(&device, &planes, experts, n, pieces);
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len().max(4) as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let inputs: Vec<u8> = (0..tokens * k)
        .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0xabcd)) - 127.0) * 0.02))
        .collect();
    let routes: Vec<u8> = (0..tokens)
        .flat_map(|t| (0..top_k).flat_map(move |s| ((t * 3 + s * 5 + t / 7) % experts).to_le_bytes()))
        .collect();
    let sorted = moe::sorted_rows(pairs, experts, &tuning).max(pairs);
    let output = u64::from(pairs) * u64::from(n) * 2;
    let rooms = [
        upload(&inputs),
        upload(&routes),
        Buffer::zeroed(&device, output).expect("y"),
        Buffer::zeroed(&device, u64::from(sorted) * 4).expect("perm"),
        Buffer::zeroed(&device, u64::from(sorted) * 4).expect("row experts"),
        Buffer::zeroed(&device, u64::from(sorted) * 4).expect("tile experts"),
        Buffer::zeroed(&device, u64::from(pairs) * 4).expect("inverse"),
        Buffer::zeroed(&device, u64::from(sorted) * u64::from(k) * 2).expect("sorted x"),
        Buffer::zeroed(&device, u64::from(sorted) * u64::from(n) * 2).expect("sorted y"),
        Buffer::zeroed(&device, u64::from(sorted) * u64::from(k / 64) * 4).expect("sums"),
        Buffer::zeroed(&device, u64::from(experts) * 4).expect("hits"),
    ];
    let at = |room: usize, rows: u32, width: u32, dtype: Dtype| {
        let buffer = &rooms[room];
        Tensor::new(
            handles.bind(buffer, 0, buffer.bytes()).expect("bind"),
            rows,
            width,
            dtype,
        )
    };
    let x = at(0, tokens, k, Dtype::Bf16);
    let r = at(1, tokens, top_k, Dtype::I32);
    let y = at(2, pairs, n, Dtype::Bf16);
    let hits = at(10, experts, 1, Dtype::U32);
    let scratch = RoutedScratch {
        perm: at(3, 1, sorted, Dtype::I32),
        row_expert: at(4, 1, sorted, Dtype::I32),
        tile_expert: at(5, 1, sorted, Dtype::I32),
        inv: at(6, 1, pairs, Dtype::I32),
        x: at(7, sorted, k, Dtype::Bf16),
        y: at(8, sorted, n, Dtype::Bf16),
        sums: at(9, sorted, k / 64, Dtype::F32),
    };
    let (bank, bias, table) = bank_of(&handles, &store, scheme, experts, n, k, pieces, Some(hits));
    let fire = |table: Option<BankTable>| {
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        if let Some(table) = table {
            moe::count_routes(&sink, r, table.hits.expect("hits")).expect("the count");
        }
        match arm {
            Arm::Qmv => match bias {
                Some(bias) => moe::matmul_select_bias(&sink, x, bank, table, bias, r, y),
                None => moe::matmul_select_quant(&sink, x, bank, table, r, y),
            }
            .expect("the routed qmv"),
            Arm::Batched(_) => {
                let batched = moe::matmul_select_batched(
                    &sink,
                    "the routed matmul",
                    x,
                    bank,
                    table,
                    bias,
                    r,
                    experts,
                    scratch,
                    y,
                    &tuning,
                    moe::Sorted::default(),
                )
                .expect("the routed matmul");
                assert!(batched, "{scheme:?} {arm:?}: these routes batch");
            }
        }
        drop(sink);
        frame.commit().expect("commit");
        handles.read(y.buf, output).expect("read")
    };
    let whole = fire(None);
    scribble(&mut store);
    let tabled = fire(Some(table));
    let counted = handles.read(hits.buf, u64::from(experts) * 4).expect("hits");
    let mut want = vec![0u32; experts as usize];
    for e in routes.chunks(4) {
        want[i32::from_le_bytes([e[0], e[1], e[2], e[3]]) as usize] += 1;
    }
    let got: Vec<u32> = counted
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got, want, "{scheme:?} {arm:?}: one count a route");
    (whole, tabled)
}

fn check(scheme: Scheme, arm: Arm, biased: bool) {
    if !engine_metal::device::present() {
        return;
    }
    if matches!(arm, Arm::Batched(_))
        && !Context::bind().expect("the system device").supports_elastic()
    {
        return;
    }
    for pieces in [1, 2] {
        let (whole, tabled) = run(scheme, arm, biased, pieces);
        assert!(whole.iter().any(|&b| b != 0), "{scheme:?} {arm:?}: rows landed");
        assert!(
            whole == tabled,
            "{scheme:?} {arm:?} biased={biased} pieces={pieces}: the tabled bank lands other bits"
        );
        eprintln!("{scheme:?} {arm:?} biased={biased} pieces={pieces}: tabled = contiguous");
    }
}

const AFFINE: [Scheme; 5] = [
    Scheme::Affine { bits: 4, group: 64 },
    Scheme::Affine { bits: 8, group: 64 },
    Scheme::Affine { bits: 2, group: 64 },
    Scheme::Affine { bits: 2, group: 32 },
    Scheme::Affine { bits: 2, group: 128 },
];

#[test]
fn the_routed_qmv_reads_a_tabled_bank_as_its_contiguous_bank() {
    for scheme in AFFINE.into_iter().chain([Scheme::Mxfp4]) {
        for biased in [false, true] {
            check(scheme, Arm::Qmv, biased);
        }
    }
}

#[test]
fn the_batched_routed_matmul_reads_a_tabled_bank_as_its_contiguous_bank() {
    // The fp16 tiled kernel (affine 4/64), the stamped tiled one (every
    // affine width), the tiled MXFP4 one with and without its bias.
    check(AFFINE[0], Arm::Batched("tiled"), false);
    for scheme in [AFFINE[0], AFFINE[1], AFFINE[2]] {
        check(scheme, Arm::Batched("stamped"), false);
    }
    for biased in [false, true] {
        check(Scheme::Mxfp4, Arm::Batched("tiled"), biased);
    }
    // The MPP kernels: affine 4/64, and MXFP4 with and without its bias.
    check(AFFINE[0], Arm::Batched("mpp"), false);
    for biased in [false, true] {
        check(Scheme::Mxfp4, Arm::Batched("mpp"), biased);
    }
}

/// The register tiles' gate-up and down of five tokens' routes (eight
/// distinct experts of 24 each) over tabled banks against the same over
/// the contiguous ones: the tiles' hidden tables and their rows, bit for
/// bit, one piece an expert and two.
#[test]
fn the_register_tiles_read_tabled_banks_as_their_contiguous_banks() {
    const EXPERTS: u32 = 24;
    const TOP_K: u32 = 8;
    const HIDDEN: u32 = 2048;
    const INTER: u32 = 512;
    const TOKENS: u32 = 5;
    if !engine_metal::device::present() {
        return;
    }
    let scheme = Scheme::Affine { bits: 4, group: 64 };
    let device = Context::bind().expect("the system device");
    let pipelines = Pipelines::new();
    for pieces in [1u32, 2] {
        let handles = Handles::new();
        let gate_planes = planes(scheme, EXPERTS, 2 * INTER, HIDDEN, false, 0x1);
        let down_planes = planes(scheme, EXPERTS, HIDDEN, INTER, false, 0x2);
        let mut gate_store = store(&device, &gate_planes, EXPERTS, 2 * INTER, pieces);
        let mut down_store = store(&device, &down_planes, EXPERTS, HIDDEN, pieces);
        // As a checkpoint's experts land: an expert a row of codes.
        let expert_rows = |bank: Bank, columns: u32| Bank {
            codes: Tensor::new(bank.codes.buf, EXPERTS, columns * bank.codes.width, bank.codes.dtype),
            ld: columns * bank.codes.width,
            ..bank
        };
        let (gate_up, _, gate_table) =
            bank_of(&handles, &gate_store, scheme, EXPERTS, 2 * INTER, HIDDEN, pieces, None);
        let (down, _, down_table) =
            bank_of(&handles, &down_store, scheme, EXPERTS, HIDDEN, INTER, pieces, None);
        let (gate_up, down) = (expert_rows(gate_up, 2 * INTER), expert_rows(down, HIDDEN));
        let plane = |bytes: u64| Buffer::zeroed(&device, bytes.max(4)).expect("a plane");
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
        let upload = |bytes: &[u8]| {
            let mut buffer = plane(bytes.len() as u64);
            buffer.write(0, bytes).expect("upload");
            buffer
        };
        let routes: Vec<i32> = (0..TOKENS)
            .flat_map(|t| (0..TOP_K).map(move |s| ((t * 5 + s * 3) % EXPERTS) as i32))
            .collect();
        let x: Vec<u8> = (0..TOKENS * HIDDEN)
            .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0x77)) - 127.0) / 64.0))
            .collect();
        let xb = upload(&x);
        let rb = upload(&routes.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>());
        let xt = Tensor::new(bind(&xb), TOKENS, HIDDEN, Dtype::Bf16);
        let rt = Tensor::new(bind(&rb), TOKENS, TOP_K, Dtype::I32);
        let tiles = moe::register_tiles(TOKENS, TOP_K, EXPERTS, false);
        let pairs = TOKENS * TOP_K;
        let room = |width: u64, dtype: Dtype, bytes: u64| {
            let b = plane(bytes);
            let t = Tensor::new(bind(&b), 1, u32::try_from(width).unwrap(), dtype);
            (b, t)
        };
        let (_a, tiles_t) = room(u64::from(tiles) + 1, Dtype::I32, (u64::from(tiles) + 1) * 4);
        let (_b, rows_t) = room(u64::from(tiles) * 8, Dtype::I32, u64::from(tiles) * 32);
        let (_c, inv_t) = room(u64::from(pairs), Dtype::I32, u64::from(pairs) * 4);
        let in_width = u64::from(tiles) * moe::register_table_width(HIDDEN);
        let (_e, table_t) = room(in_width, Dtype::Bf16, in_width * 2);
        let mid_width = u64::from(tiles) * moe::register_table_width(INTER);
        let (_f, hidden_t) = room(mid_width, Dtype::Bf16, mid_width * 2);
        let out_bytes = u64::from(tiles) * 8 * u64::from(HIDDEN) * 2;
        let (_y, y_t) = room(u64::from(tiles) * 8 * u64::from(HIDDEN), Dtype::Bf16, out_bytes);
        let (_m, mix_t) = room(
            u64::from(moe::REGISTER_MIX),
            Dtype::F32,
            u64::from(moe::REGISTER_MIX) * 4,
        );
        let (_n, grids_t) = room(
            u64::from(moe::REGISTER_GRIDS),
            Dtype::I32,
            u64::from(moe::REGISTER_GRIDS) * 4,
        );
        let rooms = moe::RegisterRooms {
            tiles: tiles_t,
            rows: rows_t,
            inv: inv_t,
            table: table_t,
            hidden: hidden_t,
            y: Tensor::new(y_t.buf, tiles * 8, HIDDEN, Dtype::Bf16),
            mix: mix_t,
            grids: grids_t,
        };
        let cut = moe::REGISTER_CUT;
        let fire = |tables: Option<(BankTable, BankTable)>| {
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            moe::register_group(&sink, xt, rt, EXPERTS, false, None, cut.groups(INTER, HIDDEN), rooms)
                .expect("group");
            moe::register_glu(
                &sink,
                gate_up,
                tables.map(|t| t.0),
                None,
                HIDDEN,
                INTER,
                tiles,
                EXPERTS,
                rooms,
                cut,
            )
            .expect("register gate-up");
            moe::register_down(
                &sink,
                down,
                tables.map(|t| t.1),
                None,
                INTER,
                HIDDEN,
                tiles,
                EXPERTS,
                rooms,
                cut,
            )
            .expect("register down");
            drop(sink);
            frame.commit().expect("commit");
            (
                handles.read(rooms.hidden.buf, mid_width * 2).expect("hidden"),
                handles.read(rooms.y.buf, out_bytes).expect("rows"),
            )
        };
        let whole = fire(None);
        scribble(&mut gate_store);
        scribble(&mut down_store);
        let tabled = fire(Some((gate_table, down_table)));
        assert!(whole.1.iter().any(|&b| b != 0), "rows landed");
        assert!(whole.0 == tabled.0, "pieces={pieces}: the gate-up lands other bits");
        assert!(whole.1 == tabled.1, "pieces={pieces}: the down lands other bits");
        eprintln!("register tiles pieces={pieces}: tabled = contiguous");
    }
}
