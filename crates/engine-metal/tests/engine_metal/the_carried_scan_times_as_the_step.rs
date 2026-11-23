#![cfg(target_vendor = "apple")]
//! Timing, run by hand (`--ignored --nocapture`): a decode lane's carried
//! scan beside the plain scan step it stands for, one lane carrying nothing,
//! Qwen3.5-0.8B's heads, each dispatched many times in one frame.

use engine_metal::chip;
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::ssm::{self, Committed, Corrected, DotsTable, Paged, VerifyGates};
use kernels_metal::tensor::RecurrentPool;
use kernels_metal::{StateMap, Tensor, Tuning};
use poem_ir::Dtype;

const K_HEADS: u32 = 16;
const V_HEADS: u32 = 16;
const D: u32 = 128;
const REPS: usize = 200;

#[test]
#[ignore = "a timing, run by hand"]
fn the_carried_scan_times_as_the_step() {
    let Ok(device) = Context::bind() else { return };
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
    let zeros = |n: usize, rows: u32, width: u32, dtype: Dtype| upload(&vec![0u8; n], rows, width, dtype);
    let ints = |v: &[i32]| upload(&v.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(), v.len() as u32, 1, Dtype::I32);
    let small = |n: usize| -> Vec<u8> { (0..n).flat_map(|i| (((i % 7) as f32) * 0.01).to_le_bytes()).collect() };
    let bf = |n: usize| -> Vec<u8> { (0..n).flat_map(|i| (((((i % 11) as f32) * 0.05).to_bits() >> 16) as u16).to_le_bytes()).collect() };
    let width = 2 * K_HEADS * D + V_HEADS * D;
    let qkv = upload(&bf(width as usize), 1, width, Dtype::Bf16);
    let bank = (V_HEADS * D * D) as usize;
    let state = upload(&small(bank), V_HEADS * D, D, Dtype::F32);
    let map = StateMap::whole(bank as u64);
    let segments = upload(&map.bytes(), 1, map.words().len() as u32, Dtype::U64);
    let pool = RecurrentPool { state, slots: ints(&[0]), conv_state: state, new_conv_state: state, segments };
    let y = zeros((V_HEADS * D * 4) as usize, 1, V_HEADS * D, Dtype::F32);
    let gates = upload(&small((2 * V_HEADS) as usize), 1, 2 * V_HEADS, Dtype::F32);

    let (profile, _) = chip::resolve(device.name(), &chip::facts(device.family(), device.cores(), 10, true), None);
    let tuning = Tuning::new(profile, false, device.cores());
    let time = |name: &str, fire: &dyn Fn(&Sink)| {
        // One warm frame, then the timed one.
        for timed in [false, true] {
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            let began = std::time::Instant::now();
            for _ in 0..REPS {
                fire(&sink);
            }
            frame.commit().expect("commit");
            if timed {
                eprintln!("{name}: {:.2} us a dispatch", began.elapsed().as_secs_f64() * 1e6 / REPS as f64);
            }
        }
    };
    time("scan step", &|sink| {
        ssm::gated_delta(sink, &tuning, qkv, qkv, gates, &pool, K_HEADS, V_HEADS, D, D, y).expect("step");
    });

    // The carried scan: one lane carrying `rep` rows through a page table
    // over four pages of each plane.
    let page_bytes = 4 * V_HEADS * D * 4;
    let slab = || zeros(page_bytes as usize * 4, 4, page_bytes, Dtype::U8);
    let (ba_slab, keys, u) = (slab(), slab(), slab());
    let mut table = vec![0i32; 11];
    for (at, slot) in table[1..5].iter_mut().enumerate() {
        *slot = at as i32;
    }
    let pages = ints(&table);
    let indptr = ints(&[0, 1]);
    let verify = VerifyGates {
        ba: upload(&bf((2 * V_HEADS) as usize), 1, 2 * V_HEADS, Dtype::Bf16),
        dt_bias: upload(&bf(V_HEADS as usize), V_HEADS, 1, Dtype::Bf16),
        a_log: upload(&small(V_HEADS as usize), V_HEADS, 1, Dtype::F32),
        gates: zeros((2 * V_HEADS * 4) as usize, 1, 2 * V_HEADS, Dtype::F32),
    };
    let ba_paged = Paged { slab: ba_slab, pages, page_tokens: 4, page_bytes, row_bytes: 2 * V_HEADS * 2, page_ints: 11 };
    // The rows' scalars in the `u` slab's pages (their values only time).
    let corrected = Corrected {
        keys,
        key_row_bytes: K_HEADS * D * 2,
        u,
        u_row_bytes: V_HEADS * D * 4,
        scalars: u,
        scalars_row_bytes: 2 * V_HEADS * 4,
    };
    for rep in [0i32, 3, 7] {
        let committed = Committed {
            replay: ints(&[rep]),
            commit: ints(&[1]),
            slots: ints(&[0]),
            lane0: 0,
            ext_replay: ints(&[0]),
        };
        let dots = DotsTable {
            table: zeros(16 * (K_HEADS as usize) * 16, 1, 4, Dtype::F32),
            rows: 16,
            max_rep: rep as u32,
        };
        time(&format!("carried scan, {rep} carried"), &|sink| {
            ssm::gated_delta_carried(
                sink, qkv, indptr, &committed, &verify, ba_paged, corrected, &pool, K_HEADS, V_HEADS, D, D, y, dots,
            )
            .expect("carried");
        });
    }

    // The conv: the prompt conv a decode row takes on the tip, beside the
    // decode window's conv over a lane carrying 0 / 3 / 7 rows.
    let channels = width;
    let taps = 4u32;
    let weight = upload(&bf((channels * taps) as usize), channels, taps, Dtype::Bf16);
    let window = (taps * channels) as usize;
    let conv_bank = upload(&small(window), taps, channels, Dtype::F32);
    let conv_map = StateMap::whole(window as u64);
    let conv_segments = upload(&conv_map.bytes(), 1, conv_map.words().len() as u32, Dtype::U64);
    let conv_pool = RecurrentPool {
        state: conv_bank,
        slots: ints(&[0]),
        conv_state: conv_bank,
        new_conv_state: conv_bank,
        segments: conv_segments,
    };
    let x = upload(&bf(channels as usize), 1, channels, Dtype::Bf16);
    let conv_y = zeros((channels * 2) as usize, 1, channels, Dtype::Bf16);
    let land = zeros((channels * 2) as usize, 1, channels, Dtype::Bf16);
    time("prompt conv", &|sink| {
        ssm::causal_conv1d(sink, x, weight, None, &conv_pool, taps, 1, conv_y).expect("conv");
    });
    let x_page = 4 * channels * 2;
    let x_slab = zeros(x_page as usize * 4, 4, x_page, Dtype::U8);
    let x_paged = Paged { slab: x_slab, pages: ints(&table), page_tokens: 4, page_bytes: x_page, row_bytes: channels * 2, page_ints: 11 };
    for rep in [0i32, 3, 7] {
        let committed = Committed {
            replay: ints(&[rep]),
            commit: ints(&[1]),
            slots: ints(&[0]),
            lane0: 0,
            ext_replay: ints(&[0]),
        };
        time(&format!("decode conv, {rep} carried"), &|sink| {
            ssm::causal_conv1d_verify(
                sink, x, Some(x_paged), indptr, &committed, weight, &conv_pool, taps, 1, conv_y, land, None, true,
            )
            .expect("decode conv");
        });
    }
}
