//! **THE SELECTOR'S TWO KERNELS, AGAINST HOST REFERENCES** — the gate for
//! `layout/topk.metal` and `attn/selector_walk.metal` (DFlash2's readout).
//!
//! 1. `topk_rows` over a wide bf16 row (a vocabulary's width, 248 320) must
//!    answer the host's sort: the same sixteen indices in the same order,
//!    values equal — with a planted tie, which must go to the LOWER column,
//!    and a planted NaN, which must never be chosen.
//! 2. `selector_walk` over two requests of different spans must pick what the
//!    reference's `lattice` + `walk_greedy` (`mlx_dspark.dflash_model.
//!    CandidateSelector`) pick, written out on the host in f32: slot by slot,
//!    `argmax_c unary[c] + Σ_r A[prev][r]·hp[r]·B[cand[c]][r]`, `prev` the
//!    anchor first and the previous pick after. The bilinear is checked to
//!    matter (a walk that ignored it would pick the unary argmax), and the
//!    second request must not read the first's anchor or picks.
//!
//! ```text
//! cargo test -p engine-metal --release --test the_selector_reads_out_the_reference_walk -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::attn::selector;
use kernels_metal::layout;
use kernels_metal::tensor::RaggedTensor;
use model_ir::Dtype;

const K: u32 = 16;
const RANK: u32 = 256;
/// A small vocabulary for the walk's codebooks; the top-k check uses the
/// real one.
const VOCAB: u32 = 4096;
const WIDE: u32 = 248_320;
const SPANS: [u32; 2] = [8, 5];

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .map(|f| ((f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16) as u16)
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

struct Dev {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
}

impl Dev {
    fn buffer(&self, bytes: &[u8]) -> (Buffer, u32) {
        let mut b = Buffer::zeroed(&self.device, bytes.len().max(4) as u64).expect("a buffer");
        if !bytes.is_empty() {
            b.write(0, bytes).expect("write");
        }
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }
    fn zeroed(&self, bytes: u64) -> (Buffer, u32) {
        let b = Buffer::zeroed(&self.device, bytes).expect("a buffer");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }
    fn run(&self, fire: &dyn Fn(&Sink)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        fire(&sink);
        frame.commit().expect("the commit");
    }
    fn read(&self, h: u32, bytes: u64) -> Vec<u8> {
        self.handles.read(h, bytes).expect("read")
    }
}

#[test]
fn topk_answers_the_host_sort_with_ties_low_and_nans_never() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let dev = Dev { device, handles: Handles::new(), pipelines: Pipelines::new() };
    let rows = 3u32;
    let mut x: Vec<f32> = (0..(rows * WIDE) as u64).map(|at| bf16_round(4.0 * unit(at))).collect();
    // Row 1: a planted tie at the top between columns 777 and 100_000 — the
    // lower column must come first. Row 2: a NaN that would win by value.
    let w = WIDE as usize;
    x[w + 100_000] = 8.0;
    x[w + 777] = 8.0;
    x[2 * w + 5] = f32::NAN;
    let (_xb, hx) = dev.buffer(&bf16_bytes(&x));
    let (_vb, hv) = dev.zeroed(u64::from(rows * K) * 4);
    let (_ib, hi) = dev.zeroed(u64::from(rows * K) * 4);
    let xt = Tensor::new(hx, rows, WIDE, Dtype::Bf16);
    let vt = Tensor::new(hv, rows, K, Dtype::F32);
    let it = Tensor::new(hi, rows, K, Dtype::I32);
    dev.run(&|sink| layout::topk(sink, xt, K, vt, it).expect("the top-k launch"));
    let values = f32s(&dev.read(hv, u64::from(rows * K) * 4));
    let indices = i32s(&dev.read(hi, u64::from(rows * K) * 4));

    for r in 0..rows as usize {
        let row = &x[r * w..(r + 1) * w];
        let mut order: Vec<usize> = (0..w).filter(|&c| !row[c].is_nan()).collect();
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        let want: Vec<i32> = order[..K as usize].iter().map(|&c| c as i32).collect();
        let got = &indices[r * K as usize..(r + 1) * K as usize];
        assert_eq!(got, &want[..], "row {r}: the indices are not the host sort's");
        for j in 0..K as usize {
            assert_eq!(values[r * K as usize + j], row[got[j] as usize], "row {r} slot {j}: value");
        }
    }
    assert_eq!(indices[K as usize], 777, "the tie went to the higher column");
    assert!(!indices[2 * K as usize..].contains(&5), "the NaN was chosen");
    eprintln!("topk: three rows of {WIDE} agree with the host sort; tie low, NaN never");
}

#[test]
fn the_walk_picks_what_the_reference_picks() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let dev = Dev { device, handles: Handles::new(), pipelines: Pipelines::new() };
    let rows: u32 = SPANS.iter().sum();
    let indptr: Vec<i32> = {
        let mut v = vec![0i32];
        for s in SPANS {
            v.push(v.last().unwrap() + s as i32);
        }
        v
    };
    let k = K as usize;
    let rank = RANK as usize;
    // Candidates: distinct ids a row; unary logits close together so the
    // bilinear decides; codebooks and hp at bf16 precision.
    let cand: Vec<i32> = (0..rows as usize)
        .flat_map(|r| (0..k).map(move |c| (noise((r * k + c) as u64) % VOCAB) as i32))
        .collect();
    let unary: Vec<f32> = (0..(rows as usize * k) as u64).map(|at| 0.05 * unit(at ^ 0x77)).collect();
    let hp: Vec<f32> = (0..(rows as usize * rank) as u64).map(|at| bf16_round(unit(at ^ 0x99))).collect();
    let pred: Vec<f32> = (0..(VOCAB as usize * rank) as u64).map(|at| bf16_round(0.2 * unit(at ^ 0xAB))).collect();
    let succ: Vec<f32> = (0..(VOCAB as usize * rank) as u64).map(|at| bf16_round(0.2 * unit(at ^ 0xCD))).collect();
    let tokens: Vec<i32> = (0..rows).map(|r| (noise(u64::from(r) ^ 0xEF) % VOCAB) as i32).collect();

    let (_cb, hc) = dev.buffer(&i32_bytes(&cand));
    let (_pb, hp_indptr) = dev.buffer(&i32_bytes(&indptr));
    let (_ub, hu) = dev.buffer(&f32_bytes(&unary));
    let (_hb, hh) = dev.buffer(&bf16_bytes(&hp));
    let (_tb, ht) = dev.buffer(&i32_bytes(&tokens));
    let (_ab, ha) = dev.buffer(&bf16_bytes(&pred));
    let (_bb, hbk) = dev.buffer(&bf16_bytes(&succ));
    let (_ob, ho) = dev.zeroed(u64::from(rows) * 4);
    let candt = RaggedTensor {
        data: Tensor::new(hc, rows, K, Dtype::I32),
        indptr: Tensor::new(hp_indptr, indptr.len() as u32, 1, Dtype::I32),
    };
    dev.run(&|sink| {
        selector::walk(
            sink,
            candt,
            Tensor::new(hu, rows, K, Dtype::F32),
            Tensor::new(hh, rows, RANK, Dtype::Bf16),
            Tensor::new(ht, rows, 1, Dtype::I32),
            Tensor::new(ha, VOCAB, RANK, Dtype::Bf16),
            Tensor::new(hbk, VOCAB, RANK, Dtype::Bf16),
            Tensor::new(ho, rows, 1, Dtype::I32),
        )
        .expect("the walk");
    });
    let picks = i32s(&dev.read(ho, u64::from(rows) * 4));

    // The reference walk, and the unary-only walk it must differ from.
    let mut want = vec![0i32; rows as usize];
    let mut unary_only = vec![0i32; rows as usize];
    let mut bilinear_mattered = false;
    for (r, &span) in SPANS.iter().enumerate() {
        let begin = indptr[r] as usize;
        want[begin] = cand[begin * k];
        unary_only[begin] = cand[begin * k];
        let mut prev = tokens[begin] as usize;
        for row in begin + 1..begin + span as usize {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            let mut best_u = 0usize;
            let mut best_uv = f32::NEG_INFINITY;
            for c in 0..k {
                let cid = cand[row * k + c] as usize;
                let mut dot = 0.0f32;
                for d in 0..rank {
                    dot += pred[prev * rank + d] * hp[row * rank + d] * succ[cid * rank + d];
                }
                let s = unary[row * k + c] + dot;
                if s > best_v {
                    best_v = s;
                    best = c;
                }
                if unary[row * k + c] > best_uv {
                    best_uv = unary[row * k + c];
                    best_u = c;
                }
            }
            want[row] = cand[row * k + best];
            unary_only[row] = cand[row * k + best_u];
            if best != best_u {
                bilinear_mattered = true;
            }
            prev = want[row] as usize;
        }
    }
    assert!(bilinear_mattered, "the fixture's bilinear never changed a pick, so the check proves nothing");
    assert_eq!(picks, want, "the walk parts from the reference");
    assert_ne!(picks, unary_only, "the walk is the unary argmax");
    eprintln!("walk: {} rows over two requests agree with the reference; the bilinear decided at least one slot", rows);
}
