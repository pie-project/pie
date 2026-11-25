//! Score capture on the first device, against an f64 host reference of what
//! the CPU oracle computes (`engine_cpu::Forward`'s capture): five lanes of
//! a fire over a paged pool with pages out of order, three of them capturing
//! (interleaved with two that do not), so the capture table compacts them.
//! Each capturing lane's one f32 row is a mass a key of the load's context:
//! a layer's causal softmax over its keys summed over the layer's query
//! heads, averaged over its last `observe` rows and the model's query heads,
//! written over a row of sevens (`into` none) and then added into by a
//! second layer (`into` some); the final row summed a kv page for the two
//! lanes that read pages. Contexts below and past a tile of keys, observe
//! windows narrower and wider than a lane's rows, and a model whose query
//! heads are twice this row's (a rank's share).

use kernels_slang::attn::score::{Capture, TABLE_INTS, stats_floats};
use kernels_slang::{KvPool, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

const PAGE: usize = 4;
const HEAD_DIM: usize = 64;
const KV_HEADS: usize = 2;
const Q_HEADS: usize = 4;
const SCALE: f64 = 0.125;

/// A lane of the fixture fire: its rows, the keys it held before them,
/// whether it captures and whether its program reads pages.
#[derive(Clone, Copy)]
struct LaneAt {
    rows: usize,
    held: usize,
    captures: bool,
    pages: bool,
}

const LANES: [LaneAt; 5] = [
    LaneAt { rows: 2, held: 3, captures: false, pages: false },
    LaneAt { rows: 6, held: 0, captures: true, pages: true },
    LaneAt { rows: 1, held: 9, captures: true, pages: false },
    LaneAt { rows: 1, held: 2, captures: false, pages: false },
    LaneAt { rows: 3, held: 0, captures: true, pages: true },
];

struct Fixture {
    lanes: Vec<LaneAt>,
    qo: Vec<i32>,
    positions: Vec<i32>,
    page_indptr: Vec<u32>,
    page_indices: Vec<u32>,
    pool_pages: usize,
    /// The two layers' queries and keys, as the device holds them.
    q: [Vec<f64>; 2],
    k: [Vec<f64>; 2],
}

fn fixture(context: usize) -> Fixture {
    // The last lane decodes near the end of the context.
    let mut lanes = LANES.to_vec();
    lanes[4].held = context - 3 - lanes[4].rows;
    let mut qo = vec![0i32];
    let mut positions = Vec::new();
    let mut page_indptr = vec![0u32];
    let mut pages_of = Vec::new();
    for lane in &lanes {
        qo.push(qo.last().unwrap() + lane.rows as i32);
        positions.extend((lane.held..lane.held + lane.rows).map(|p| p as i32));
        let pages = (lane.held + lane.rows).div_ceil(PAGE);
        page_indptr.push(page_indptr.last().unwrap() + pages as u32);
        pages_of.push(pages);
    }
    let pool_pages = *page_indptr.last().unwrap() as usize;
    // Pages out of order: a stride prime to the pool's pages.
    let step = (3..).find(|s| pool_pages % s != 0).unwrap();
    let page_indices = (0..pool_pages).map(|i| ((i * step + 1) % pool_pages) as u32).collect();
    let rows = *qo.last().unwrap() as usize;
    let slots = pool_pages * PAGE;
    let widen = |v: Vec<f32>| -> Vec<f64> {
        v.into_iter()
            .map(|x| f64::from(crate::gpu::widen(crate::gpu::bf16(x))))
            .collect()
    };
    Fixture {
        lanes,
        qo,
        positions,
        page_indptr,
        page_indices,
        pool_pages,
        q: [
            widen(noise(2, rows * Q_HEADS * HEAD_DIM)),
            widen(noise(3, rows * Q_HEADS * HEAD_DIM)),
        ],
        k: [
            widen(noise(4, slots * KV_HEADS * HEAD_DIM)),
            widen(noise(5, slots * KV_HEADS * HEAD_DIM)),
        ],
    }
}

/// Each capturing lane's key row after both layers, and its page row.
fn reference(f: &Fixture, context: usize, observe: usize, q_heads: usize) -> Vec<Vec<f64>> {
    let row = KV_HEADS * HEAD_DIM;
    let mut out = Vec::new();
    for (lane, _) in f.lanes.iter().enumerate().filter(|(_, at)| at.captures) {
        let mut keys = vec![0.0f64; context];
        let pages = &f.page_indices[f.page_indptr[lane] as usize..f.page_indptr[lane + 1] as usize];
        let capacity = pages.len() * PAGE;
        let (lo, hi) = (f.qo[lane] as usize, f.qo[lane + 1] as usize);
        let observed = observe.min(hi - lo);
        for layer in 0..2 {
            for h in 0..Q_HEADS {
                let kh = h / (Q_HEADS / KV_HEADS);
                for qi in hi - observed..hi {
                    let limit = (f.positions[qi] as usize + 1).min(capacity);
                    let qv = &f.q[layer][(qi * Q_HEADS + h) * HEAD_DIM..][..HEAD_DIM];
                    let s: Vec<f64> = (0..limit)
                        .map(|j| {
                            let slot = pages[j / PAGE] as usize * PAGE + j % PAGE;
                            let kv = &f.k[layer][slot * row + kh * HEAD_DIM..][..HEAD_DIM];
                            qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f64>() * SCALE
                        })
                        .collect();
                    let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = s.iter().map(|v| (v - top).exp()).sum();
                    for (j, v) in s.iter().enumerate().take(context) {
                        keys[j] += (v - top).exp() / total / (observed * q_heads) as f64;
                    }
                }
            }
        }
        out.push(keys);
    }
    out
}

/// Whether `got` is `want` to f32 accumulation.
fn agrees(got: &[f32], want: &[f64]) -> Result<(), String> {
    for (at, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-6 + w.abs() * 1e-4;
        let err = (f64::from(*g) - w).abs();
        if err > tol {
            return Err(format!("at {at}: got {g}, want {w} (err {err}, tol {tol})"));
        }
    }
    Ok(())
}

/// The capture table the engine stages: lane, rows, and the bases of its
/// key row and page row.
fn table(f: &Fixture, context: usize) -> Vec<i32> {
    let mut page_row = 0;
    let pages = context.div_ceil(PAGE) as i32;
    f.lanes
        .iter()
        .enumerate()
        .filter(|(_, at)| at.captures)
        .enumerate()
        .flat_map(|(row, (lane, at))| {
            let page = if at.pages {
                page_row += 1;
                (page_row - 1) * pages
            } else {
                -1
            };
            [lane as i32, f.qo[lane], f.qo[lane + 1], (row * context) as i32, page]
        })
        .collect()
}

/// `workgroup_bytes`: the workgroup memory the device is taken to state,
/// which picks the mass kernel's form.
fn case(gpu: &mut Gpu, context: usize, observe: usize, q_heads: usize, workgroup_bytes: u32) {
    let f = fixture(context);
    let what = format!(
        "{workgroup_bytes} B of workgroup memory: context {context}, observe {observe}, {q_heads} model heads"
    );
    let rows = *f.qo.last().unwrap() as usize;
    let slots = f.pool_pages * PAGE;
    let row = KV_HEADS * HEAD_DIM;
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let up_bf16 = |gpu: &mut Gpu, v: &[f64]| {
        gpu.up(&v.iter().map(|x| *x as f32).collect::<Vec<_>>()).0
    };
    let pi = gpu.up_u32(&f.page_indices);
    let pp = gpu.up_u32(&f.page_indptr);
    let pos = gpu.up_i32(&f.positions);
    let capture_table = table(&f, context);
    let capturing = (capture_table.len() / TABLE_INTS as usize) as u32;
    let page_rows = capture_table.chunks(TABLE_INTS as usize).filter(|e| e[4] >= 0).count();
    let tb = gpu.up_i32(&capture_table);
    let pools: Vec<KvPool> = (0..2)
        .map(|layer| {
            let keys = up_bf16(gpu, &f.k[layer]);
            KvPool::planes(
                t(keys, slots, row, Dtype::Bf16),
                t(keys, slots, row, Dtype::Bf16),
                t(pi, f.page_indices.len(), 1, Dtype::U32),
                t(pp, f.page_indptr.len(), 1, Dtype::U32),
                PAGE as i32,
                row as u64,
                HEAD_DIM as u64,
            )
        })
        .collect();
    let qs: Vec<u32> = (0..2).map(|layer| up_bf16(gpu, &f.q[layer])).collect();
    // One row past the capturing lanes', which no capture writes.
    let held_rows = capturing as usize + 1;
    let scores = gpu.up_f32(&vec![7.0; held_rows * context]);
    let stats_n = stats_floats(capturing, Q_HEADS as u32, observe as u32) as usize;
    let stats = gpu.up_f32(&vec![0.0; stats_n]);
    let pages_w = context.div_ceil(PAGE);
    let pages = gpu.up_f32(&vec![7.0; (page_rows + 1) * pages_w]);
    let keys_t = t(scores, capturing as usize, context, Dtype::F32);
    gpu.run(|c| {
        for layer in 0..2 {
            kernels_slang::attn::score::capture(
                c,
                &Capture {
                    q: t(qs[layer], rows, Q_HEADS * HEAD_DIM, Dtype::Bf16),
                    positions: t(pos, rows, 1, Dtype::I32),
                    table: t(tb, capturing as usize, TABLE_INTS as usize, Dtype::I32),
                    capturing,
                    stats: t(stats, 1, stats_n, Dtype::F32),
                    scores: keys_t,
                },
                &pools[layer],
                HEAD_DIM as u32,
                KV_HEADS as u32,
                q_heads as u32,
                SCALE as f32,
                observe as u32,
                layer > 0,
                workgroup_bytes,
            )?;
        }
        kernels_slang::attn::score::pages(
            c,
            t(tb, capturing as usize, TABLE_INTS as usize, Dtype::I32),
            capturing,
            keys_t,
            t(pages, page_rows, pages_w, Dtype::F32),
            PAGE as u32,
        )
    });

    let got = gpu.down_f32(scores, held_rows * context);
    let want = reference(&f, context, observe, q_heads);
    for (c, keys) in want.iter().enumerate() {
        agrees(&got[c * context..][..context], keys)
            .unwrap_or_else(|why| panic!("{what}: capturing lane {c}'s key row: {why}"));
        // Each layer's mass is the lane's share of the model's heads.
        let mass: f64 = got[c * context..][..context].iter().map(|v| f64::from(*v)).sum();
        let share = 2.0 * Q_HEADS as f64 / q_heads as f64;
        assert!((mass - share).abs() < 1e-4, "{what}: lane {c} holds {mass}, not {share}");
    }
    assert!(
        got[capturing as usize * context..].iter().all(|&v| v == 7.0),
        "{what}: the row past the capturing lanes is no capture's"
    );

    let got_pages = gpu.down_f32(pages, (page_rows + 1) * pages_w);
    let mut p = 0;
    for (c, at) in f.lanes.iter().filter(|at| at.captures).enumerate() {
        if !at.pages {
            continue;
        }
        let summed: Vec<f64> = want[c].chunks(PAGE).map(|page| page.iter().sum()).collect();
        agrees(&got_pages[p * pages_w..][..pages_w], &summed)
            .unwrap_or_else(|why| panic!("{what}: page row {p} (lane {c}): {why}"));
        p += 1;
    }
    assert!(
        got_pages[page_rows * pages_w..].iter().all(|&v| v == 7.0),
        "{what}: the page row past the readers' is no sum's"
    );
}

#[test]
fn score_capture_answers_the_cpu_oracles_sum() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    for workgroup_bytes in [16 << 10, 32 << 10] {
        for (context, observe, q_heads) in [(32, 2, 4), (40, 5, 8), (300, 1, 4), (300, 32, 8)] {
            case(&mut gpu, context, observe, q_heads, workgroup_bytes);
        }
    }
}
