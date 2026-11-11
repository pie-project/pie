#![cfg(target_vendor = "apple")]

//! Every paged-kv kernel finds a token's row at `plane + offset + page *
//! page_stride + token * row`. Run once over the dense pool (each plane its own
//! buffer of `page_size`-row pages, today's layout) and once over one buffer
//! holding both planes page by page, each page's keys and values at their own
//! offsets with poison between and around them (the shape a layer-major page
//! takes), the append lands the same rows, every read route lands the same
//! output and log-sum-exp bits, and the poison stays.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{DecodePlan, KvPool, PrefillPlan, RaggedTensor, Tensor, attn};
use poem_ir::Dtype;

const POISON: u8 = 0xa5;
/// Bytes of poison before, between and after a page's two planes.
const GAP: u64 = 256;

fn vals(n: usize, s: u64) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ s;
            x ^= x >> 33;
            x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
            let f = ((x >> 40) as f32 / 16777216. - 0.5) * 4.;
            let b = f.to_bits();
            (((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Rig {
    dev: Context,
    handles: Handles,
    pipes: Pipelines,
    keep: std::cell::RefCell<Vec<Buffer>>,
}

impl Rig {
    fn upload(&self, v: &[u8]) -> u32 {
        let mut b = Buffer::zeroed(&self.dev, (v.len() as u64).max(16)).unwrap();
        b.write(0, v).unwrap();
        let h = self.handles.bind(&b, 0, b.bytes()).unwrap();
        self.keep.borrow_mut().push(b);
        h
    }

    /// `v` uploaded: a handle on the whole buffer, and one bound at each of
    /// `offsets` into it (a plane bound at its rows of page 0).
    fn upload_at(&self, v: &[u8], offsets: &[u64]) -> (u32, Vec<u32>) {
        let mut b = Buffer::zeroed(&self.dev, (v.len() as u64).max(16)).unwrap();
        b.write(0, v).unwrap();
        let whole = self.handles.bind(&b, 0, b.bytes()).unwrap();
        let at = offsets
            .iter()
            .map(|&offset| self.handles.bind(&b, offset, b.bytes() - offset).unwrap())
            .collect();
        self.keep.borrow_mut().push(b);
        (whole, at)
    }

    fn tensor(&self, v: &[u8], rows: usize, width: usize, dtype: Dtype) -> Tensor {
        Tensor::new(self.upload(v), rows as u32, width as u32, dtype)
    }

    fn read(&self, h: u32, len: u64) -> Vec<u8> {
        self.handles.read(h, len).unwrap()
    }

    fn run(&self, f: &dyn Fn(&Sink)) {
        let frame = self.dev.frame().unwrap();
        f(&Sink::new(&self.dev, &frame, &self.pipes, &self.handles));
        frame.commit().unwrap();
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Bf16,
    Q8,
    U4,
}

#[test]
fn a_scattered_kv_pool_is_the_dense_one() {
    let Ok(dev) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let rig = Rig {
        dev,
        handles: Handles::new(),
        pipes: Pipelines::new(),
        keep: std::cell::RefCell::new(Vec::new()),
    };
    // (kind, head width, query rows, query heads, kv heads, requests, pages a
    // request, first position, page size)
    for config in [
        (
            Kind::Bf16,
            64usize,
            35usize,
            6usize,
            2usize,
            3usize,
            3usize,
            65usize,
            32usize,
        ),
        (Kind::Bf16, 128, 35, 6, 2, 3, 3, 65, 32),
        (Kind::Bf16, 256, 35, 6, 2, 3, 3, 65, 32),
        (Kind::Bf16, 512, 35, 6, 2, 3, 3, 65, 32),
        (Kind::Bf16, 256, 32, 24, 4, 4, 5, 65, 32),
        (Kind::Bf16, 128, 35, 6, 2, 3, 6, 65, 16),
        (Kind::Q8, 128, 35, 6, 2, 3, 3, 65, 32),
        (Kind::Q8, 256, 35, 6, 2, 3, 3, 65, 32),
        (Kind::Q8, 256, 32, 24, 4, 4, 5, 65, 32),
        (Kind::U4, 128, 35, 6, 2, 3, 3, 65, 32),
        (Kind::U4, 256, 35, 6, 2, 3, 6, 65, 16),
    ] {
        agree(&rig, config);
    }
}

#[allow(clippy::type_complexity)]
fn agree(
    rig: &Rig,
    (kind, d, m, qh, kh, requests, np, base, page): (
        Kind,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
) {
    let pages = np * requests;
    let slots = pages * page;
    let dtype = if kind == Kind::U4 {
        Dtype::KvU4
    } else {
        Dtype::Bf16
    };
    let row = (kh * if kind == Kind::U4 { d / 2 + 2 } else { 2 * d }) as u64;
    let page_bytes = row * page as u64;
    let k = vals(slots * kh * d, 7);
    let v = vals(slots * kh * d, 19);
    let q = vals(m * qh * d, 41);
    let owners: Vec<u32> = (0..m).map(|i| (i % requests) as u32).collect();
    let pos: Vec<u32> = (0..m).map(|i| (base + i / requests) as u32).collect();
    let table: Vec<u32> = (0..pages as u32).rev().collect();
    let ptr: Vec<u32> = (0..=requests).map(|i| (i * np) as u32).collect();
    let wp: Vec<u32> = (0..slots).map(|i| (i / page) as u32).collect();
    let wo: Vec<u32> = (0..slots).map(|i| (i % page) as u32).collect();

    let page_indices = rig.tensor(&u32s(&table), pages, 1, Dtype::U32);
    let page_indptr = rig.tensor(&u32s(&ptr), requests + 1, 1, Dtype::U32);
    let plane = |h: u32| Tensor::new(h, slots as u32, (kh * d) as u32, dtype);
    let base_pool = |keys: u32, values: u32| KvPool {
        keys: plane(keys),
        values: plane(values),
        page_indices,
        page_indptr,
        page_size: page as i32,
        seq_stride: (kh * d) as u64,
        head_stride: d as u64,
        q8: kind == Kind::Q8,
        page_stride: [0; 2],
    };
    // Today's pool: a buffer a plane, pages packed from its origin.
    let dense_k = rig.upload(&vec![POISON; (page_bytes * pages as u64) as usize]);
    let dense_v = rig.upload(&vec![POISON; (page_bytes * pages as u64) as usize]);
    let dense = base_pool(dense_k, dense_v).paged_densely();
    assert_eq!(dense.page_stride, [page_bytes, page_bytes]);
    // One buffer: page `p`'s keys at `GAP + p * stride`, its values a page
    // and a gap later, a gap after them; each plane bound at its rows of
    // page 0.
    let stride = 2 * page_bytes + 3 * GAP;
    let slab_bytes = stride * pages as u64 + GAP;
    let (slab, planes) = rig.upload_at(
        &vec![POISON; slab_bytes as usize],
        &[GAP, 2 * GAP + page_bytes],
    );
    let scattered = KvPool {
        page_stride: [stride, stride],
        ..base_pool(planes[0], planes[1])
    };

    let what = format!("{kind:?} d={d} m={m} page={page}");
    let kt = rig.tensor(&k, slots, kh * d, Dtype::Bf16);
    let vt = rig.tensor(&v, slots, kh * d, Dtype::Bf16);
    let wpt = rig.tensor(&u32s(&wp), slots, 1, Dtype::U32);
    let wot = rig.tensor(&u32s(&wo), slots, 1, Dtype::U32);
    for pool in [&dense, &scattered] {
        rig.run(&|s| attn::kv_append(s, kt, vt, pool, wpt, wot).unwrap());
    }
    // The append: each page's rows where the layout puts them, poison around.
    let dk = rig.read(dense_k, page_bytes * pages as u64);
    let dv = rig.read(dense_v, page_bytes * pages as u64);
    let mut want = vec![POISON; slab_bytes as usize];
    for p in 0..pages {
        let at = (p as u64 * stride) as usize;
        let span = (p as u64 * page_bytes) as usize..((p + 1) as u64 * page_bytes) as usize;
        let k_at = at + GAP as usize;
        let v_at = at + (2 * GAP + page_bytes) as usize;
        want[k_at..k_at + page_bytes as usize].copy_from_slice(&dk[span.clone()]);
        want[v_at..v_at + page_bytes as usize].copy_from_slice(&dv[span]);
    }
    assert!(
        rig.read(slab, slab_bytes) == want,
        "{what}: the scattered append is not the dense one placed"
    );

    let qt = rig.tensor(&q, m, qh * d, Dtype::Bf16);
    let qp = rig.tensor(&u32s(&[0, m as u32]), 2, 1, Dtype::I32);
    let positions = rig.tensor(&u32s(&pos), m, 1, Dtype::I32);
    let request_of_token = rig.tensor(&u32s(&owners), m, 1, Dtype::I32);
    let scale = 1. / (d as f32).sqrt();
    let mut routes = 0;
    for (causal, window, mode) in [
        (true, None, 0u8),
        (true, Some(19), 1),
        (true, None, 2),
        (true, None, 3),
        (false, Some(19), 1),
    ] {
        let mask: Vec<u8> = (0..m * np * page)
            .map(|i| if mode == 3 { 0 } else { u8::from(i % 7 != 0) })
            .collect();
        let mask_t = rig.tensor(&mask, mask.len(), 1, Dtype::U8);
        let enabled = rig.tensor(&vec![if mode == 3 { 1 } else { mode }; m], m, 1, Dtype::U8);
        let plan = DecodePlan {
            positions,
            request_of_token,
            mask: mask_t,
            mask_enabled: enabled,
            mask_stride: (np * page) as u32,
        };
        let pre = PrefillPlan {
            positions,
            last_page_len: positions,
            request_of_token,
            mask: mask_t,
            mask_enabled: enabled,
            mask_stride: plan.mask_stride,
        };
        let per =
            attn::split::workspace_words(attn::split::split_count(m as u32, 32), qh as u32, 256)
                .unwrap();
        let words = m.div_ceil(8) * per as usize;
        for route in 0..7 {
            let packed = kind == Kind::U4;
            // The packed pool reads through the decode and the tiled prompt.
            if packed && !matches!(route, 0 | 3) || !causal && route != 1 && route != 3 {
                continue;
            }
            if route == 4 && (d != 256 || page != 32 || !(8..=128).contains(&m) || window.is_some())
            {
                continue;
            }
            if route >= 5 && (packed || !causal) {
                continue;
            }
            let mut outs = Vec::new();
            for pool in [&dense, &scattered] {
                let y = rig.tensor(&vec![POISON; q.len() + 64], m, qh * d, Dtype::Bf16);
                let lse = rig.tensor(&vec![POISON; m * qh * 4 + 64], m, qh, Dtype::F32);
                let workspace = rig.tensor(&vec![POISON; words * 4 + 64], 1, words, Dtype::F32);
                let ragged = RaggedTensor {
                    data: qt,
                    indptr: qp,
                };
                let tuning = engine_metal::chip::tuning("", false);
                let mpp = engine_metal::chip::tuning("[kernel]\nsdpa.prefill = \"mpp\"", false);
                rig.run(&|s| match route {
                    0 => attn::decode(s, &tuning, qt, &plan, pool, window, d as u32, scale, y)
                        .unwrap(),
                    1 => attn::arbiter::masked(
                        s,
                        ragged,
                        &pre,
                        plan.mask,
                        pool,
                        window,
                        causal,
                        d as u32,
                        scale,
                        y,
                        requests as u32,
                        &mpp,
                    )
                    .unwrap(),
                    2 => attn::decode_lse(
                        s, &tuning, qt, &plan, pool, window, d as u32, scale, y, lse,
                    )
                    .unwrap(),
                    3 => attn::prefill(
                        s, &tuning, ragged, &pre, pool, window, d as u32, kh as u32, scale, y,
                    )
                    .unwrap(),
                    4 => assert!(
                        attn::split::try_paged(
                            s,
                            qt,
                            pool,
                            &pre,
                            plan.mask,
                            window,
                            causal,
                            d as u32,
                            scale,
                            y,
                            requests as u32,
                            &|rows, width| (rows == 1 && width as usize <= words)
                                .then_some(Tensor { width, ..workspace }),
                            32,
                            kind == Kind::Bf16,
                        )
                        .unwrap()
                    ),
                    // The prompt's own arbitration: the simdgroup-matrix
                    // kernels where they serve, the tiled read else.
                    5 => attn::arbiter::masked(
                        s,
                        ragged,
                        &pre,
                        plan.mask,
                        pool,
                        window,
                        causal,
                        d as u32,
                        scale,
                        y,
                        requests as u32,
                        &tuning,
                    )
                    .unwrap(),
                    _ => attn::arbiter::masked_lse(
                        s,
                        ragged,
                        &pre,
                        plan.mask,
                        pool,
                        window,
                        causal,
                        d as u32,
                        scale,
                        y,
                        lse,
                        requests as u32,
                        &tuning,
                    )
                    .unwrap(),
                });
                outs.push((
                    rig.read(y.buf, (q.len() + 64) as u64),
                    rig.read(lse.buf, (m * qh * 4 + 64) as u64),
                ));
            }
            assert!(
                outs[0].0 == outs[1].0,
                "{what}: route {route} mode {mode} causal {causal}: the scattered pool's \
                 output is not the dense one's"
            );
            assert!(
                outs[0].1 == outs[1].1,
                "{what}: route {route} mode {mode}: the scattered pool's log-sum-exp is not \
                 the dense one's"
            );
            assert!(
                outs[0].0[..q.len()] != vec![POISON; q.len()][..],
                "{what}: route {route} landed nothing"
            );
            routes += 1;
        }
    }
    assert!(
        rig.read(slab, slab_bytes) == want,
        "{what}: a read wrote the pool"
    );
    eprintln!("{what}: the append and {routes} read routes agree bit for bit");
}
