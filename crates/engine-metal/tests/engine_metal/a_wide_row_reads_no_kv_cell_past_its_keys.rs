#![cfg(target_vendor = "apple")]
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{KvPool, PrefillPlan, RaggedTensor, Tensor, attn};
use poem_ir::Dtype;

/// A row's mask words, one u32 a row (its enable flag, its boundary 0).
fn mask_words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn bf(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
fn vals(n: usize, s: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ s;
            x ^= x >> 33;
            x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
            bf(((x >> 40) as f32 / 16777216. - 0.5) * 4.)
        })
        .collect()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect()
}

/// A wide row (mask mode 2: it reads every key its lane holds that its mask
/// admits) over a lane whose keys end inside its last page: the cells past
/// them were never written, and the kv pool's memory is recycled, so they may
/// hold a NaN. Here they do, and the mask admits none of them. Each prefill
/// form (mpp, mma, tiled; causal and not, which the arbiter routes as it
/// would) must keep them out of the row, by bounds or by select, not by a zero
/// weight: 0 x NaN is NaN.
#[test]
fn a_wide_row_reads_no_kv_cell_past_its_keys() {
    let Ok(dev) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipes = Pipelines::new();
    let mut wrong = Vec::new();
    for (form, tuning) in [
        ("mpp", "[kernel]\nsdpa.prefill = \"mpp\""),
        ("mma", "[kernel]\nsdpa.prefill = \"tiled\"\nsdpa.mma = true"),
        (
            "tiled",
            "[kernel]\nsdpa.prefill = \"tiled\"\nsdpa.mma = false",
        ),
    ] {
        let tuning = engine_metal::chip::tuning(tuning, false);
        for (d, page) in [(128usize, 32usize), (128, 16), (64, 32), (64, 16)] {
            // The configurations a served model stages: a wide row (mode 2) only
            // in a class that attends both ways, so not causal; a plain or masked
            // row causal, its keys ending at its own position.
            for (mode, causal) in [(2u32, false), (1, true), (0, true)] {
                // Rows enough a lane that each form tiles (not the vector walk): two
                // lanes of 48 rows over 96 keys' pages.
                let (m, qh, kh, requests) = (96usize, 4usize, 2usize, 2usize);
                let np = 96 / page;
                // Each lane holds `held` keys, ending inside its last page.
                let held = np * page - 11;
                let slots = np * requests * page;
                let pages: Vec<u32> = (0..(np * requests) as u32).rev().collect();
                let ptr: Vec<u32> = (0..=requests).map(|i| (i * np) as u32).collect();
                let unwritten = |slot: usize| {
                    let (page_id, cell) = (slot / page, slot % page);
                    let at = pages.iter().position(|&p| p as usize == page_id).unwrap();
                    (at % np) * page + cell >= held
                };
                let mut k = vals(slots * kh * d, 7);
                let mut v = vals(k.len(), 19);
                for slot in (0..slots).filter(|&s| unwritten(s)) {
                    for x in &mut k[slot * kh * d..(slot + 1) * kh * d] {
                        *x = f32::NAN;
                    }
                    for x in &mut v[slot * kh * d..(slot + 1) * kh * d] {
                        *x = f32::NAN;
                    }
                }
                let q = vals(m * qh * d, 41);
                let owners: Vec<u32> = (0..m).map(|i| (i % requests) as u32).collect();
                let pos: Vec<u32> = (0..m).map(|i| (held - 1 - i / requests) as u32).collect();
                let upload = |v: &[u8]| {
                    let mut b = Buffer::zeroed(&dev, v.len() as u64).unwrap();
                    b.write(0, v).unwrap();
                    b
                };
                let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).unwrap();
                let tensor = |b: &Buffer, r: usize, c: usize, t: Dtype| {
                    Tensor::new(bind(b), r as u32, c as u32, t)
                };
                let (kp, vp, qb) = (upload(&bytes(&k)), upload(&bytes(&v)), upload(&bytes(&q)));
                let y = upload(&vec![0xa5; q.len() * 2 + 64]);
                let (pb, ip, po, ow) = (
                    upload(&u32s(&pages)),
                    upload(&u32s(&ptr)),
                    upload(&u32s(&pos)),
                    upload(&u32s(&owners)),
                );
                let qp = upload(&u32s(&[0, m as u32]));
                let pool = KvPool {
                    keys: tensor(&kp, slots, kh * d, Dtype::Bf16),
                    values: tensor(&vp, slots, kh * d, Dtype::Bf16),
                    page_indices: tensor(&pb, np * requests, 1, Dtype::U32),
                    page_indptr: tensor(&ip, requests + 1, 1, Dtype::U32),
                    page_size: page as i32,
                    seq_stride: (kh * d) as u64,
                    head_stride: d as u64,
                    q8: false,
                    page_stride: [0; 2],
                }
                .paged_densely();
                // Every held key admitted, none past them.
                let mask: Vec<u8> = (0..m * np * page)
                    .map(|i| u8::from(i % (np * page) < held))
                    .collect();
                let mb = upload(&mask);
                let eb = upload(&mask_words(&vec![mode; m]));
                let plan = PrefillPlan {
                    positions: tensor(&po, m, 1, Dtype::I32),
                    last_page_len: tensor(&po, m, 1, Dtype::I32),
                    request_of_token: tensor(&ow, m, 1, Dtype::I32),
                    mask: tensor(&mb, mask.len(), 1, Dtype::U8),
                    mask_enabled: tensor(&eb, m, 1, Dtype::U32),
                    mask_stride: (np * page) as u32,
                };
                let scale = 1. / (d as f32).sqrt();
                let frame = dev.frame().unwrap();
                let ran = attn::arbiter::masked(
                    &Sink::new(&dev, &frame, &pipes, &handles),
                    RaggedTensor {
                        data: tensor(&qb, m, qh * d, Dtype::Bf16),
                        indptr: tensor(&qp, 2, 1, Dtype::I32),
                    },
                    &plan,
                    plan.mask,
                    &pool,
                    None,
                    causal,
                    d as u32,
                    scale,
                    tensor(&y, m, qh * d, Dtype::Bf16),
                    requests as u32,
                    &tuning,
                );
                if let Err(_) = ran {
                    continue;
                }
                frame.commit().unwrap();
                let got = f32s(&handles.read(bind(&y), y.bytes()).unwrap()[..q.len() * 2]);
                let bad = got.iter().filter(|x| !x.is_finite()).count();
                if bad > 0 {
                    wrong.push(format!(
                        "{form} d={d} page={page} mode={mode} causal={causal}: {bad} non-finite"
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "a wide row read a kv cell past its keys: {wrong:#?}"
    );
}
