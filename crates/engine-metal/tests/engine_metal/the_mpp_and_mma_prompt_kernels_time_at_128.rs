#![cfg(target_vendor = "apple")]
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{KvPool, PrefillPlan, RaggedTensor, Tensor, attn};
use poem_ir::Dtype;

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

/// Times a 128-wide causal prompt through the MPP kernel against the form
/// `sdpa.prefill = "tiled"` takes (the simdgroup-matrix kernel at 128), over
/// a bf16 pool of 32-token pages (the only pages MPP serves), at 2048 and
/// 8190 rows, Qwen3-0.6B's heads (16 query, 8 kv). Prints the best GPU time
/// of three fires each and how far the two outputs lie apart. Ignored: a
/// measurement for the M5 (where MPP runs on the tensor units) to decide
/// whether MPP is worth bringing to the 16-token pages serving builds.
///
///     cargo test --release -p engine-metal --test engine_metal \
///         the_mpp_and_mma_prompt_kernels_time_at_128 -- --ignored --nocapture
#[test]
#[ignore = "a timing for the M5, run by hand"]
fn the_mpp_and_mma_prompt_kernels_time_at_128() {
    let Ok(dev) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipes = Pipelines::new();
    let forms = [
        ("mpp", engine_metal::chip::tuning("[kernel]\nsdpa.prefill = \"mpp\"", false)),
        ("tiled (mma at 128)", engine_metal::chip::tuning("[kernel]\nsdpa.prefill = \"tiled\"", false)),
    ];
    let (d, qh, kh, page) = (128usize, 16usize, 8usize, 32usize);
    for m in [2048usize, 8190] {
        let np = m.div_ceil(page);
        let slots = np * page;
        let k = vals(slots * kh * d, 7);
        let v = vals(k.len(), 19);
        let q = vals(m * qh * d, 41);
        let pos: Vec<u32> = (0..m as u32).collect();
        let pages: Vec<u32> = (0..np as u32).rev().collect();
        let upload = |v: &[u8]| {
            let mut b = Buffer::zeroed(&dev, v.len() as u64).unwrap();
            b.write(0, v).unwrap();
            b
        };
        let kp = upload(&bytes(&k));
        let vp = upload(&bytes(&v));
        let qb = upload(&bytes(&q));
        let pb = upload(&u32s(&pages));
        let ip = upload(&u32s(&[0, np as u32]));
        let po = upload(&u32s(&pos));
        let ow = upload(&u32s(&vec![0; m]));
        let qp = upload(&u32s(&[0, m as u32]));
        let mb = upload(&[1u8]);
        let eb = upload(&vec![0u8; m]);
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).unwrap();
        let tensor =
            |b: &Buffer, r: usize, c: usize, t: Dtype| Tensor::new(bind(b), r as u32, c as u32, t);
        let pool = KvPool {
            keys: tensor(&kp, slots, kh * d, Dtype::Bf16),
            values: tensor(&vp, slots, kh * d, Dtype::Bf16),
            page_indices: tensor(&pb, np, 1, Dtype::U32),
            page_indptr: tensor(&ip, 2, 1, Dtype::U32),
            page_size: page as i32,
            seq_stride: (kh * d) as u64,
            head_stride: d as u64,
            q8: false,
            page_stride: [0; 2],
        }
        .paged_densely();
        let plan = PrefillPlan {
            positions: tensor(&po, m, 1, Dtype::I32),
            last_page_len: tensor(&po, m, 1, Dtype::I32),
            request_of_token: tensor(&ow, m, 1, Dtype::I32),
            mask: tensor(&mb, 1, 1, Dtype::U8),
            mask_enabled: tensor(&eb, m, 1, Dtype::U8),
            mask_stride: 0,
        };
        let scale = 1. / (d as f32).sqrt();
        let mut outs = Vec::new();
        for (name, tuning) in &forms {
            let form = attn::arbiter::prefill_form(d as u32, page as i32, tuning);
            let y = upload(&vec![0u8; q.len() * 2]);
            let mut best = f64::INFINITY;
            for _ in 0..4 {
                let frame = dev.frame().unwrap();
                attn::arbiter::prefill(
                    &Sink::new(&dev, &frame, &pipes, &handles),
                    RaggedTensor {
                        data: tensor(&qb, m, qh * d, Dtype::Bf16),
                        indptr: tensor(&qp, 2, 1, Dtype::I32),
                    },
                    &plan,
                    &pool,
                    None,
                    d as u32,
                    kh as u32,
                    scale,
                    tensor(&y, m, qh * d, Dtype::Bf16),
                    1,
                    tuning,
                )
                .unwrap();
                best = best.min(frame.commit_timed().unwrap());
            }
            eprintln!("d=128 rows={m}: {name} ({form:?}) best of 4: {:.2} ms", best * 1e3);
            outs.push(f32s(&handles.read(bind(&y), y.bytes()).unwrap()));
        }
        let apart = outs[0]
            .iter()
            .zip(&outs[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("d=128 rows={m}: the two outputs lie at most {apart:.4} apart");
    }
}
