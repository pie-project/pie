#![cfg(target_vendor = "apple")]
//! A decode lane carrying more rows than the carried scan's per-head table
//! holds (a beam whose lineages stay apart, a long run-ahead) is served, not
//! refused: the scan makes each chunk's reductions itself, and answers as it
//! does through the table.

use engine_metal::chip;
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::ssm::{self, Committed, Corrected, DotsTable, Paged, VerifyGates};
use kernels_metal::tensor::RecurrentPool;
use kernels_metal::{StateMap, Tensor, Tuning};
use poem_ir::Dtype;

const K_HEADS: u32 = 2;
const V_HEADS: u32 = 2;
const D: u32 = 128;
const CARRIED: u32 = 40;
const PAGE: u32 = 4;
const PAGES: u32 = 12;

fn noise(at: u64) -> f32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5EED;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn bf16_bytes(n: usize, salt: u64, scale: f32) -> Vec<u8> {
    (0..n).flat_map(|i| (((noise(salt ^ i as u64) * scale).to_bits() >> 16) as u16).to_le_bytes()).collect()
}

fn f32_bytes(n: usize, salt: u64, scale: f32) -> Vec<u8> {
    (0..n).flat_map(|i| (noise(salt ^ i as u64) * scale).to_le_bytes()).collect()
}

#[test]
fn a_carry_past_the_table_is_served() {
    let Ok(device) = Context::bind() else { return };
    let (profile, _) = chip::resolve(device.name(), &chip::facts(device.family(), device.cores(), 10, true), None);
    let tuning = Tuning::new(profile, false, device.cores());
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let keep = std::cell::RefCell::new(Vec::new());
    let upload = |bytes: &[u8], rows: u32, width: u32, dtype: Dtype| -> Tensor {
        let mut b = Buffer::zeroed(&device, (bytes.len() as u64).max(16)).expect("buffer");
        b.write(0, bytes).expect("write");
        let h = handles.bind(&b, 0, b.bytes()).expect("bind");
        keep.borrow_mut().push(b);
        Tensor::new(h, rows, width, dtype)
    };
    let ints = |v: &[i32]| upload(&v.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(), v.len() as u32, 1, Dtype::I32);
    let read = |t: Tensor, n: usize| -> Vec<f32> {
        handles.read(t.buf, n as u64 * 4).expect("read").chunks_exact(4).map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect()
    };
    let width = 2 * K_HEADS * D + V_HEADS * D;
    let bank = (V_HEADS * D * D) as usize;
    let map = StateMap::whole(bank as u64);
    let segments = upload(&map.bytes(), 1, map.words().len() as u32, Dtype::U64);
    let qkv = upload(&bf16_bytes(width as usize, 0x11, 1.0), 1, width, Dtype::Bf16);
    // A page of each plane a slab slot; the lane's rows from token 0.
    let page_bytes = PAGE * V_HEADS * D * 4;
    let slab = |row_bytes: usize, salt: u64, f32_rows: bool| {
        let mut bytes = vec![0u8; (PAGES * page_bytes) as usize];
        for row in 0..(PAGES * PAGE) as usize {
            let at = (row / PAGE as usize) * page_bytes as usize + (row % PAGE as usize) * row_bytes;
            let data = if f32_rows {
                f32_bytes(row_bytes / 4, salt ^ ((row as u64) << 20), 0.3)
            } else {
                bf16_bytes(row_bytes / 2, salt ^ ((row as u64) << 20), 1.0)
            };
            bytes[at..at + row_bytes].copy_from_slice(&data);
        }
        upload(&bytes, PAGES, page_bytes, Dtype::U8)
    };
    let (ba_row, key_row, u_row) = ((2 * V_HEADS * 2) as usize, (K_HEADS * D * 2) as usize, (V_HEADS * D * 4) as usize);
    let (ba_slab, key_slab, u_slab) = (slab(ba_row, 0x22, false), slab(key_row, 0x33, false), slab(u_row, 0x44, true));
    let ints_a_row = 1 + PAGES;
    let mut table = vec![0i32; ints_a_row as usize];
    for (at, slot) in table[1..].iter_mut().enumerate() {
        *slot = at as i32;
    }
    let pages = ints(&table);
    let indptr = ints(&[0, 1]);
    let committed = Committed {
        replay: ints(&[CARRIED as i32]),
        commit: ints(&[3]),
        slots: ints(&[0]),
        lane0: 0,
        ext_replay: ints(&[0]),
    };
    let verify = VerifyGates {
        ba: upload(&bf16_bytes((2 * V_HEADS) as usize, 0x55, 2.0), 1, 2 * V_HEADS, Dtype::Bf16),
        dt_bias: upload(&bf16_bytes(V_HEADS as usize, 0x66, 1.0), V_HEADS, 1, Dtype::Bf16),
        a_log: upload(&f32_bytes(V_HEADS as usize, 0x77, 1.0), V_HEADS, 1, Dtype::F32),
        gates: upload(&vec![0u8; (2 * V_HEADS * 4) as usize], 1, 2 * V_HEADS, Dtype::F32),
    };
    let ba_paged = Paged { slab: ba_slab, pages, page_tokens: PAGE, page_bytes, row_bytes: ba_row as u32, page_ints: ints_a_row };
    // A long carry takes the staged scan, which derives its rows' scalars:
    // the own row lands its in a plane of its own.
    let scalars = upload(&vec![0u8; (PAGES * page_bytes) as usize], PAGES, page_bytes, Dtype::U8);
    let corrected = Corrected {
        keys: key_slab,
        key_row_bytes: key_row as u32,
        u: u_slab,
        u_row_bytes: u_row as u32,
        scalars,
        scalars_row_bytes: 2 * V_HEADS * 4,
    };
    let work = upload(&vec![0u8; 64 * 1024], 1, 4, Dtype::F32);
    // The scan through a table that holds the carry, then through one that
    // does not: each from the same state.
    let mut answers = Vec::new();
    for rows in [64u32, 8] {
        let state = upload(&f32_bytes(bank, 0x88, 0.1), V_HEADS * D, D, Dtype::F32);
        let pool = RecurrentPool { state, slots: ints(&[0]), conv_state: state, new_conv_state: state, segments };
        let y = upload(&vec![0u8; (V_HEADS * D * 4) as usize], 1, V_HEADS * D, Dtype::F32);
        let dots = DotsTable { table: work, rows, max_rep: CARRIED };
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        ssm::gated_delta_carried(&sink, &tuning, qkv, indptr, &committed, &verify, ba_paged, corrected, &pool, K_HEADS, V_HEADS, D, D, y, dots)
            .expect("a carry past the table is served");
        frame.commit().expect("commit");
        answers.push((read(y, (V_HEADS * D) as usize), read(state, bank)));
    }
    let (table_y, table_state) = &answers[0];
    let (inline_y, inline_state) = &answers[1];
    let apart = |a: &[f32], b: &[f32]| {
        let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale
    };
    let (dy, ds) = (apart(table_y, inline_y), apart(table_state, inline_state));
    eprintln!("readout {dy:e}, state {ds:e} of their scales apart");
    assert!(dy < 1e-4 && ds < 1e-4, "the in-scan reductions answer as the table: readout {dy:e}, state {ds:e}");
}
