#![cfg(feature = "cuda")]

//! The paged decode (FA2 and the tensor-core decode, under each fold of a
//! split) and a one-row-a-request prefill answer the host softmax.

use crate::common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn;
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;
const INT_BYTES: usize = 8 << 20;
const FLOAT_BYTES: usize = 128 << 20;

/// One decode fire: its requests' kv lengths and its heads.
#[derive(Clone, Copy)]
struct Case {
    lens: &'static [usize],
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    window: Option<u32>,
    /// Whether the plan is the one a recorded body replays (split kv, padded).
    graph: bool,
}

struct Fixture {
    gpu: Gpu,
    case: Case,
    pool: KvPool,
    kv_indptr: Vec<i32>,
    kv_len: Vec<i32>,
    q: RaggedTensor,
    o: u64,
    lse: u64,
    /// gpt-oss's per-head sinks, f32.
    sinks: u64,
    host_sinks: Vec<f32>,
    workspace: Workspace,
    host_q: Vec<f32>,
    host_k: Vec<f32>,
    host_v: Vec<f32>,
    /// Each request's pages, in order.
    pages: Vec<Vec<usize>>,
    /// The fire's live seat, `[rows, row offset, lanes, lane offset]`, as
    /// a recorded body stages it for a plan over some of its lanes.
    stage: u64,
}

fn fixture(case: Case, seed: u64) -> Fixture {
    let mut gpu = Gpu::open();
    let width = (case.kv_heads * case.head_dim) as usize;
    let mut kv_indptr = vec![0i32];
    let mut pages = Vec::new();
    let mut next = 1usize;
    for &len in case.lens {
        let n = len.div_ceil(PAGE);
        // Each request's pages run backwards through its stretch of the
        // pool, so no page follows its neighbour.
        pages.push((0..n).map(|i| next + n - 1 - i).collect::<Vec<_>>());
        next += n;
        kv_indptr.push(kv_indptr.last().unwrap() + n as i32);
    }
    let cells = next * PAGE;
    let mut lcg = Lcg::seeded(seed);
    let (k_bits, host_k) = lcg.row(cells * width);
    let (v_bits, host_v) = lcg.row(cells * width);
    let rows = case.lens.len();
    let q_width = (case.q_heads * case.head_dim) as usize;
    let (q_bits, host_q) = lcg.row(rows * q_width);
    let page_indices: Vec<i32> = pages.iter().flatten().map(|&p| p as i32).collect();
    let last: Vec<i32> = case
        .lens
        .iter()
        .map(|&len| ((len - 1) % PAGE + 1) as i32)
        .collect();
    let kv_len: Vec<i32> = case.lens.iter().map(|&l| l as i32).collect();
    let qo_indptr: Vec<i32> = (0..=rows as i32).collect();
    let keys = gpu.up(&k_bits);
    let values = gpu.up(&v_bits);
    let pool = KvPool {
        keys: Tensor::new(keys, cells as u32, width as u32, Dtype::Bf16),
        values: Tensor::new(values, cells as u32, width as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(
            gpu.up(&page_indices),
            page_indices.len() as u32,
            1,
            Dtype::I32,
        ),
        page_indptr: Tensor::new(gpu.up(&kv_indptr), kv_indptr.len() as u32, 1, Dtype::I32),
        last_page_lens: Tensor::new(gpu.up(&last), last.len() as u32, 1, Dtype::I32),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: width as i64,
        head_stride: i64::from(case.head_dim),
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: pages.iter().map(Vec::len).max().unwrap_or(0) as i32,
        pages_in_batch: page_indices.len() as i32,
    };
    let q = RaggedTensor {
        data: Tensor::new(gpu.up(&q_bits), rows as u32, q_width as u32, Dtype::Bf16),
        indptr: Tensor::new(gpu.up(&qo_indptr), rows as u32 + 1, 1, Dtype::I32),
    };
    let o = gpu.zeros(rows * q_width * 2);
    let lse = gpu.zeros(rows * case.q_heads as usize * 4);
    let host_sinks: Vec<f32> = (0..case.q_heads)
        .map(|h| 0.5 * (h % 7) as f32 - 1.0)
        .collect();
    let sinks = gpu.up(&host_sinks);
    let stage = gpu.zeros(16);
    let workspace = Workspace {
        int_ptr: gpu.zeros(INT_BYTES),
        int_bytes: INT_BYTES,
        float_ptr: gpu.zeros(FLOAT_BYTES),
        float_bytes: FLOAT_BYTES,
    };
    Fixture {
        gpu,
        case,
        pool,
        kv_indptr,
        kv_len,
        q,
        o,
        lse,
        sinks,
        host_sinks,
        workspace,
        host_q,
        host_k,
        host_v,
        pages,
        stage,
    }
}

impl Fixture {
    /// Plans and fires one decode (with its log-sum-exp) under `tuning`.
    fn fire(&self, tuning: Tuning) -> impl Fn() + '_ {
        self.fire_with(tuning, false)
    }

    /// [`Fixture::fire`], with the sinks landed over its output where
    /// `sinked`.
    fn fire_with(&self, tuning: Tuning, sinked: bool) -> impl Fn() + '_ {
        self.fire_lanes(tuning, sinked, 0..self.case.lens.len() as u32)
    }

    /// [`Fixture::fire_with`] over the fire's `lanes` alone (one query row
    /// a lane), its live seat staged as a recorded body stages it where they
    /// are not all of the fire's.
    fn fire_lanes(
        &self,
        tuning: Tuning,
        sinked: bool,
        lanes: core::ops::Range<u32>,
    ) -> impl Fn() + '_ {
        let ctx = self.gpu.ctx().with_tuning(tuning);
        let device = Device::probe(&ctx).expect("the device answers");
        let all = self.case.lens.len() as u32;
        let (first, count) = (lanes.start, lanes.len() as u32);
        if count != all {
            let seat = [count, first, count, first];
            self.gpu.sync();
            let legacy = common::legacy();
            let copied = unsafe {
                kernels_cuda::cudarc::runtime::sys::cudaMemcpy(
                    self.stage as *mut core::ffi::c_void,
                    seat.as_ptr().cast(),
                    16,
                    kernels_cuda::cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            assert_eq!(
                copied,
                kernels_cuda::cudarc::runtime::sys::cudaError::cudaSuccess
            );
            drop(legacy);
            ctx.arm_stage(self.stage);
        }
        let shape = Shape {
            num_requests: count,
            lane_offset: first,
            num_q_heads: self.case.q_heads,
            num_kv_heads: self.case.kv_heads,
            head_dim: self.case.head_dim,
            page_size: PAGE as u32,
            hnd_layout: false,
        };
        let live = Live {
            requests: count,
            lane_offset: first,
            row_offset: first,
            rows: count,
        };
        let (at, end) = (first as usize, (first + count) as usize);
        let max_grid = attn::fa2::decode_max_grid_size(
            self.case.head_dim,
            self.case.q_heads,
            self.case.kv_heads,
            &device,
        );
        let planned = plan::plan_decode(
            &self.kv_indptr[at..=end],
            &self.kv_len[at..end],
            shape,
            live,
            self.case.window,
            self.case.graph,
            max_grid,
            Toggles::of(tuning),
            &device,
            self.workspace,
        )
        .expect("the decode plans");
        planned.stage(&ctx).expect("the plan uploads");
        let scale = (self.case.head_dim as f32).sqrt().recip();
        let q_width = self.case.q_heads * self.case.head_dim;
        move || {
            let mut o = Tensor::new(self.o, all, q_width, Dtype::Bf16);
            let mut lse = Tensor::new(self.lse, all, self.case.q_heads, Dtype::F32);
            let sinks = Tensor::new(self.sinks, self.case.q_heads, 1, Dtype::F32);
            if sinked {
                attn::decode_lse_sink(
                    &ctx,
                    self.q,
                    &planned,
                    &self.pool,
                    self.case.window,
                    self.case.head_dim,
                    scale,
                    &mut o,
                    &mut lse,
                    sinks,
                )
            } else {
                attn::decode_lse(
                    &ctx,
                    self.q,
                    &planned,
                    &self.pool,
                    self.case.window,
                    self.case.head_dim,
                    scale,
                    &mut o,
                    &mut lse,
                )
            }
            .expect("the decode fires");
        }
    }

    /// The same rows through the tensor-core prefill kernel, one query row a
    /// request.
    fn fire_prefill(&self, tuning: Tuning) -> impl Fn() + '_ {
        self.fire_prefill_with(tuning, false)
    }

    fn fire_prefill_with(&self, tuning: Tuning, sinked: bool) -> impl Fn() + '_ {
        let ctx = self.gpu.ctx().with_tuning(tuning);
        let device = Device::probe(&ctx).expect("the device answers");
        let rows = self.case.lens.len() as u32;
        let shape = Shape {
            num_requests: rows,
            lane_offset: 0,
            num_q_heads: self.case.q_heads,
            num_kv_heads: self.case.kv_heads,
            head_dim: self.case.head_dim,
            page_size: PAGE as u32,
            hnd_layout: false,
        };
        let live = Live {
            requests: rows,
            lane_offset: 0,
            row_offset: 0,
            rows,
        };
        let qo_indptr: Vec<i32> = (0..=rows as i32).collect();
        let planned = plan::plan_prefill(
            &qo_indptr,
            &self.kv_indptr,
            &self.kv_len,
            rows,
            shape,
            live,
            self.case.window,
            true,
            self.case.graph,
            None,
            &device,
            Toggles::of(tuning),
            self.workspace,
        )
        .expect("the prefill plans");
        planned.stage(&ctx).expect("the plan uploads");
        let scale = (self.case.head_dim as f32).sqrt().recip();
        let q_width = self.case.q_heads * self.case.head_dim;
        move || {
            let mut o = Tensor::new(self.o, rows, q_width, Dtype::Bf16);
            let mut lse = Tensor::new(self.lse, rows, self.case.q_heads, Dtype::F32);
            let sinks = Tensor::new(self.sinks, self.case.q_heads, 1, Dtype::F32);
            if sinked {
                attn::prefill_lse_sink(
                    &ctx,
                    self.q,
                    &planned,
                    &self.pool,
                    self.case.window,
                    self.case.head_dim,
                    self.case.kv_heads,
                    scale,
                    &mut o,
                    &mut lse,
                    sinks,
                )
            } else {
                attn::prefill_lse(
                    &ctx,
                    self.q,
                    &planned,
                    &self.pool,
                    self.case.window,
                    self.case.head_dim,
                    self.case.kv_heads,
                    scale,
                    &mut o,
                    &mut lse,
                )
            }
            .expect("the prefill fires");
        }
    }

    /// Fills the output planes with garbage, so a fire that lands nothing
    /// is caught rather than read as the last one's answer.
    fn scrub(&self) {
        use kernels_cuda::cudarc::runtime::sys as rt;
        let rows = self.case.lens.len();
        let _legacy = common::legacy();
        unsafe {
            rt::cudaMemset(
                self.o as *mut core::ffi::c_void,
                0x7f,
                rows * (self.case.q_heads * self.case.head_dim) as usize * 2,
            );
            rt::cudaMemset(
                self.lse as *mut core::ffi::c_void,
                0x7f,
                rows * self.case.q_heads as usize * 4,
            );
        }
    }

    fn output(&self) -> (Vec<f32>, Vec<f32>) {
        self.gpu.sync();
        let rows = self.case.lens.len();
        let o: Vec<u16> = self.gpu.down(
            self.o,
            rows * (self.case.q_heads * self.case.head_dim) as usize,
        );
        (
            o.into_iter().map(common::from_bf16).collect(),
            self.gpu.down(self.lse, rows * self.case.q_heads as usize),
        )
    }

    /// The reference output: each head's softmax over the request's keys
    /// in its window, against its values, and the base-2 log-sum-exp.
    fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let (hd, kvh, qh) = (
            self.case.head_dim as usize,
            self.case.kv_heads as usize,
            self.case.q_heads as usize,
        );
        let width = kvh * hd;
        let group = qh / kvh;
        let scale = (hd as f32).sqrt().recip();
        let mut o = vec![0f32; self.case.lens.len() * qh * hd];
        let mut lse = vec![0f32; self.case.lens.len() * qh];
        for (r, &len) in self.case.lens.iter().enumerate() {
            let start = self
                .case
                .window
                .map_or(0, |w| len.saturating_sub(w as usize));
            let cell = |t: usize| self.pages[r][t / PAGE] * PAGE + t % PAGE;
            for h in 0..qh {
                let kh = h / group;
                let q = &self.host_q[(r * qh + h) * hd..][..hd];
                let scores: Vec<f32> = (start..len)
                    .map(|t| {
                        let k = &self.host_k[cell(t) * width + kh * hd..][..hd];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                lse[r * qh + h] = (max + sum.ln()) * std::f32::consts::LOG2_E;
                let out = &mut o[(r * qh + h) * hd..][..hd];
                for (s, t) in scores.iter().zip(start..len) {
                    let p = (s - max).exp() / sum;
                    let v = &self.host_v[cell(t) * width + kh * hd..][..hd];
                    for (acc, x) in out.iter_mut().zip(v) {
                        *acc += p * x;
                    }
                }
            }
        }
        (o, lse)
    }
}

fn worst(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0, f32::max)
}

/// The kernels a case reads under: FA2 or the tensor-core decode, each
/// under both folds of a split's partials (the staged persistent merge and
/// the warp-a-row one), and the tensor-core decode reading a window of up
/// to 256 keys whole or a full row across a cluster.
const READS: [(bool, bool, u32, u32); 9] = [
    (false, false, 0, 0),
    (true, false, 0, 0),
    (false, true, 0, 0),
    (true, true, 0, 0),
    (false, true, 256, 0),
    (true, true, 256, 0),
    (false, true, 0, 8),
    (true, true, 256, 2),
    (false, true, 0, 1),
];

fn check(case: Case, via_prefill: bool) {
    let fx = fixture(case, 0x5eed);
    let (want_o, want_lse) = fx.reference();
    for (merge_warp, decode_xqa, whole, cluster) in READS {
        let tuning = Tuning {
            merge_warp,
            decode_xqa,
            decode_xqa_whole_window: whole,
            decode_xqa_cluster: cluster,
            ..Tuning::DEFAULT
        };
        // Twice off one staged plan, as a step's layers read it: what one
        // fire leaves in the plan's workspace must not spoil the next.
        if via_prefill {
            let fire = fx.fire_prefill(tuning);
            fire();
            fx.scrub();
            fire();
        } else {
            let fire = fx.fire(tuning);
            fire();
            fx.scrub();
            fire();
        }
        let (o, lse) = fx.output();
        fx.scrub();
        let (eo, el) = (worst(&o, &want_o), worst(&lse, &want_lse));
        assert!(
            eo < 2e-2 && el < 1e-2,
            "{} q / {} kv heads at {}, window {:?}, graph {}, via prefill {via_prefill}, warp \
             merge {merge_warp}, tensor-core decode {decode_xqa}, whole to {whole}, cluster {cluster}: output off by \
             {eo}, lse by {el}",
            case.q_heads,
            case.kv_heads,
            case.head_dim,
            case.window,
            case.graph
        );
    }
}

const LENS: &[usize] = &[1, 15, 16, 17, 130, 300, 701];

const HEADS: [(u32, u32, u32); 4] = [(64, 8, 64), (32, 16, 256), (32, 4, 512), (8, 8, 128)];

/// Every group and head width, whole and windowed (a window a page
/// multiple would not cut), on a plan of its own and on the padded one a
/// recorded body replays; a one-row-a-request prefill, which a grouped
/// decode may read its cache through, answers the same reference.
#[test]
fn a_decode_answers_the_reference_at_each_group_and_width() {
    for (q_heads, kv_heads, head_dim) in HEADS {
        for window in [None, Some(200)] {
            for graph in [false, true] {
                let case = Case {
                    lens: LENS,
                    q_heads,
                    kv_heads,
                    head_dim,
                    window,
                    graph,
                };
                check(case, false);
                check(case, true);
            }
        }
    }
}

/// A recorded decode plan over some of a fire's lanes lands those lanes' rows
/// (sinks and all) at the plane's own rows, under every read, and leaves the
/// other lanes' rows as they were: a whole-window read lands each row where
/// its query sits, a split's fold at the staged seat.
#[test]
fn a_plan_over_some_of_a_fires_lanes_lands_their_rows_alone() {
    for window in [None, Some(128)] {
        let case = Case {
            lens: LENS,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            window,
            graph: true,
        };
        let fx = fixture(case, 0x1a7e);
        let (mut want_o, want_lse) = fx.reference();
        for (at, chunk) in want_o.chunks_mut(64).enumerate() {
            let r = 1.0
                / (1.0 + (-(want_lse[at] * std::f32::consts::LN_2 - fx.host_sinks[at % 64])).exp());
            for x in chunk {
                *x *= r;
            }
        }
        let width = 64 * 64;
        for lanes in [2..5u32, 5..7] {
            for (merge_warp, decode_xqa, whole, cluster) in READS {
                let tuning = Tuning {
                    merge_warp,
                    decode_xqa,
                    decode_xqa_whole_window: whole,
                    decode_xqa_cluster: cluster,
                    ..Tuning::DEFAULT
                };
                fx.scrub();
                fx.fire_lanes(tuning, true, lanes.clone())();
                let (o, lse) = fx.output();
                let (from, to) = (lanes.start as usize, lanes.end as usize);
                let label = format!(
                    "window {window:?}, lanes {lanes:?}, warp merge {merge_warp}, tensor-core \
                     decode {decode_xqa}, whole to {whole}, cluster {cluster}"
                );
                let eo = worst(
                    &o[from * width..to * width],
                    &want_o[from * width..to * width],
                );
                let el = worst(&lse[from * 64..to * 64], &want_lse[from * 64..to * 64]);
                assert!(
                    eo < 2e-2 && el < 1e-2,
                    "{label}: output off by {eo}, lse by {el}"
                );
                let scrubbed = common::from_bf16(0x7f7f);
                let others = (0..LENS.len()).filter(|r| !lanes.contains(&(*r as u32)));
                for r in others {
                    assert!(
                        o[r * width..(r + 1) * width].iter().all(|x| *x == scrubbed),
                        "{label}: row {r}, outside the plan's lanes, was written"
                    );
                }
            }
        }
    }
}

/// A sinked decode or prefill lands what the attention and the sink kernel
/// land after it: in the warp fold where the profile takes it, else by the
/// sink kernel, against the host softmax scaled by each head's sink.
#[test]
fn a_sinked_attention_answers_the_reference_scaled_by_its_sinks() {
    for window in [None, Some(128)] {
        for graph in [false, true] {
            let case = Case {
                lens: LENS,
                q_heads: 64,
                kv_heads: 8,
                head_dim: 64,
                window,
                graph,
            };
            let fx = fixture(case, 0x51c);
            let (mut want_o, want_lse) = fx.reference();
            for (at, chunk) in want_o.chunks_mut(64).enumerate() {
                let h = at % 64;
                let r = 1.0
                    / (1.0 + (-(want_lse[at] * std::f32::consts::LN_2 - fx.host_sinks[h])).exp());
                for x in chunk {
                    *x *= r;
                }
            }
            for (merge_warp, decode_xqa, whole, cluster) in READS {
                let tuning = Tuning {
                    merge_warp,
                    decode_xqa,
                    decode_xqa_whole_window: whole,
                    decode_xqa_cluster: cluster,
                    ..Tuning::DEFAULT
                };
                for via_prefill in [false, true] {
                    if via_prefill {
                        fx.fire_prefill_with(tuning, true)();
                    } else {
                        fx.fire_with(tuning, true)();
                    }
                    let (o, lse) = fx.output();
                    fx.scrub();
                    let (eo, el) = (worst(&o, &want_o), worst(&lse, &want_lse));
                    assert!(
                        eo < 2e-2 && el < 1e-2,
                        "window {window:?}, graph {graph}, warp merge {merge_warp}, tensor-core \
                         decode {decode_xqa}, whole to {whole}, cluster {cluster}, via prefill {via_prefill}: output \
                         off by {eo}, lse by {el}"
                    );
                }
            }
        }
    }
}
