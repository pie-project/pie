#![cfg(feature = "cuda")]

//! A captured masked prefill's body is one for every key split: each fire
//! restages its plan, the chunks its own, and a replay of one chunk (rows
//! landed whole, the fold folding none) and of many land what the eager
//! plan lands.

use crate::common::attn_ref::{pool, ten, wide};
use crate::common::host::upload_to;
use crate::common::{Gpu, Lcg, captured, forget, replay};
use dtype::Dtype::{Bf16, F32, I32, I64, U8};
use kernels_cuda::attn;
use kernels_cuda::attn::plan::{self, Device, Live, PrefillPlan, Shape, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;
/// The carve's rows: one lane.
const ROWS: usize = 256;
/// The most keys a fire here reads.
const KEYS: usize = 4096;

/// A fire: its new rows over that many cached keys before them.
#[derive(Clone, Copy, Debug)]
struct Fire {
    new: usize,
    cached: usize,
}

impl Fire {
    fn keys(self) -> usize {
        self.new + self.cached
    }
    fn pages(self) -> usize {
        self.keys().div_ceil(PAGE)
    }
}

/// Query heads, kv heads, head width and window.
#[derive(Clone, Copy, Debug)]
struct Heads {
    qh: usize,
    kvh: usize,
    d: usize,
    window: Option<u32>,
}

fn shape(h: Heads) -> Shape {
    Shape {
        num_requests: 1,
        lane_offset: 0,
        num_q_heads: h.qh as u32,
        num_kv_heads: h.kvh as u32,
        head_dim: h.d as u32,
        page_size: PAGE as u32,
        hnd_layout: false,
    }
}

fn live(fire: Fire) -> Live {
    Live {
        requests: 1,
        lane_offset: 0,
        row_offset: 0,
        rows: fire.new as u32,
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_of(
    h: Heads,
    fire: Fire,
    graph: bool,
    mask_indptr: kernels_cuda::tensor::Tensor,
    device: &Device,
    ws: Workspace,
) -> PrefillPlan {
    let qo = [0, fire.new as i32];
    let kv_indptr = [0, fire.pages() as i32];
    let kv_len = [fire.keys() as i32];
    plan::plan_prefill(
        &qo,
        &kv_indptr,
        &kv_len,
        if graph { ROWS as u32 } else { fire.new as u32 },
        shape(h),
        live(fire),
        h.window,
        true,
        graph,
        Some(mask_indptr),
        device,
        ws,
    )
    .unwrap_or_else(|e| panic!("{fire:?} plans (graph {graph}): {e}"))
}

#[test]
fn a_captured_prefill_replays_whole_and_split_as_eager() {
    for h in [
        Heads { qh: 16, kvh: 4, d: 128, window: None },
        Heads { qh: 16, kvh: 1, d: 512, window: Some(128) },
    ] {
        replays(h);
    }
}

#[allow(clippy::too_many_lines)]
fn replays(h: Heads) {
    let Heads { qh, kvh, d, .. } = h;
    // Few live tiles over many keys split; a fire whose keys are one chunk's
    // least width lands whole.
    let split = Fire {
        new: 64,
        cached: KEYS - 64,
    };
    let whole = Fire {
        new: 100,
        cached: 20,
    };
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx().with_tuning(Tuning::DEFAULT);
    let device = Device::probe(&ctx).expect("the device answers");
    let (width, qw) = (kvh * d, qh * d);
    let cells = KEYS.div_ceil(PAGE) * PAGE;
    let mut lcg = Lcg::seeded(0x5917);
    let keys = ten(gpu.up(&lcg.row(cells * width).0), cells, width, Bf16);
    let values = ten(gpu.up(&lcg.row(cells * width).0), cells, width, Bf16);
    let q_at = gpu.zeros(ROWS * qw * 2);
    let (o_at, e_at) = (gpu.zeros(ROWS * qw * 2), gpu.zeros(ROWS * qw * 2));
    let pages_at = gpu.zeros(KEYS.div_ceil(PAGE) * 4);
    let (indptr_at, last_at, qo_at) = (gpu.zeros(8), gpu.zeros(4), gpu.zeros(8));
    let bits = ROWS * KEYS;
    let mask_at = gpu.zeros(bits.div_ceil(8));
    let mask_ip_at = gpu.zeros(16);
    let mask_ip = ten(mask_ip_at, 2, 1, I64);
    let workspace = |gpu: &mut Gpu| Workspace {
        int_ptr: gpu.zeros(16 << 20),
        int_bytes: 16 << 20,
        float_ptr: gpu.zeros(64 << 20),
        float_bytes: 64 << 20,
    };
    let (ws_graph, ws_eager) = (workspace(&mut gpu), workspace(&mut gpu));
    let kvp = KvPool {
        values,
        last_page_lens: ten(last_at, 1, 1, I32),
        head_stride: d as i64,
        ..pool(keys, ten(pages_at, KEYS.div_ceil(PAGE), 1, I32), ten(indptr_at, 2, 1, I32), PAGE)
    };
    let qt = RaggedTensor {
        data: ten(q_at, ROWS, qw, Bf16),
        indptr: ten(qo_at, 2, 1, I32),
    };
    let mask = ten(mask_at, bits.div_ceil(8), 1, U8);
    let scale = (d as f32).sqrt().recip();
    // A fire's inputs, fresh: its rows, its pages (scattered), its mask.
    let mut seat = |gpu: &Gpu, fire: Fire| {
        gpu.sync();
        upload_to(q_at, &lcg.row(ROWS * qw).0);
        let pages: Vec<i32> = (0..fire.pages() as i32)
            .map(|p| (p * 37) % KEYS.div_ceil(PAGE) as i32)
            .collect();
        upload_to(pages_at, &pages);
        upload_to(indptr_at, &[0, fire.pages() as i32]);
        upload_to(last_at, &[((fire.keys() - 1) % PAGE + 1) as i32]);
        upload_to(qo_at, &[0, fire.new as i32]);
        let mask: Vec<u8> = (0..bits.div_ceil(8))
            .map(|i| if (i * 7) % 5 == 0 { 0x3f } else { 0xfd })
            .collect();
        upload_to(mask_at, &mask);
        upload_to(mask_ip_at, &[0i64, (fire.new * fire.keys()) as i64]);
    };
    let (lse_at, lse_e) = (gpu.zeros(ROWS * qh * 4), gpu.zeros(ROWS * qh * 4));
    let lse_of = |o: u64| if o == o_at { lse_at } else { lse_e };
    let fire = |plan: &PrefillPlan, o: u64| {
        attn::masked_lse(
            &ctx,
            qt,
            plan,
            mask,
            &kvp,
            h.window,
            d as u32,
            scale,
            &mut ten(o, ROWS, qw, Bf16),
            &mut ten(lse_of(o), ROWS, qh, F32),
        )
    };

    seat(&gpu, split);
    let first = plan_of(h, split, true, mask_ip, &device, ws_graph);
    assert!(first.info.split_kv, "the captured layout splits: {:?}", first.info);
    first.stage(&ctx).expect("the plan uploads");
    gpu.lay(&ctx, |_| fire(&first, o_at));
    fire(&first, o_at).expect("the warm fire");
    gpu.sync();
    let mut fired = None;
    let exec = captured(&ctx, || fired = Some(fire(&first, o_at)));
    fired.expect("the capture ran").expect("the arm fires");

    for (pass, at) in [split, whole, split, whole].into_iter().enumerate() {
        seat(&gpu, at);
        let eager = plan_of(h, at, false, mask_ip, &device, ws_eager);
        eager.stage(&ctx).expect("the eager plan uploads");
        gpu.laid(&ctx, |_| fire(&eager, e_at)).expect("the eager twin");
        let graph = plan_of(h, at, true, mask_ip, &device, ws_graph);
        assert_eq!(graph.info, first.info, "one layout for every replay");
        graph.stage(&ctx).expect("the plan uploads");
        replay(&ctx, exec);
        gpu.sync();
        let word = graph.info.split_live_offset.expect("a split layout's word");
        let staged: Vec<i32> = gpu.down(ws_graph.int_ptr + u64::from(word), 1);
        eprintln!("{h:?}: replay {pass} of {at:?} splits: {}", staged[0]);
        if h.window.is_none() {
            assert_eq!(
                staged[0] == 1,
                at.keys() > 128,
                "replay {pass} of {at:?} splits its keys where they hold two chunks"
            );
        }
        let rows = at.new * qw;
        let got: Vec<u16> = gpu.down(o_at, rows);
        let want: Vec<u16> = gpu.down(e_at, rows);
        let what = format!("{h:?}: replay {pass} of {at:?}");
        let (gl, wl): (Vec<f32>, Vec<f32>) =
            (gpu.down(lse_at, at.new * qh), gpu.down(lse_e, at.new * qh));
        let lworst = gl.iter().zip(&wl).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        eprintln!("{what}: lse worst {lworst} off the eager plan");
        assert!(lworst <= 2e-2, "{what}: lse {lworst} off the eager plan");
        if staged[0] == 0 {
            // One chunk, the eager plan's one too: the same sums.
            assert_eq!(got, want, "{what} lands the eager bits");
        } else {
            let (got, want) = (wide(&got), wide(&want));
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            eprintln!("{what}: worst {worst} off the eager plan");
            assert!(worst <= 2e-2, "{what}: {worst} off the eager plan");
        }
    }
    forget(exec);
}
