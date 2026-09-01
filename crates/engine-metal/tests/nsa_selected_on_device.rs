//! The dsv4 NSA fine branch end to end, on a real Apple GPU: the indexer's
//! ranking, and the pooled reader that walks what it chose.
//!
//! **WHAT THIS FILE CLOSES.** `attention.index_topk` has been served on this
//! plane since the index family landed, and every gate on it so far handed
//! the selected reader a HAND-WRITTEN selection row — a rectangle the test
//! typed, not one a kernel produced. Nothing measured the two ends against
//! each other, so nothing could have caught a ranking that published ids in
//! one space and a reader that walked another. That is exactly the mistake
//! this family was one step away from making: dsv4-flash's indexer keys one
//! row per COMPRESSED BLOCK, not one per token, so its ids are compressed
//! rows and the cell each names is `(c + 1) * ratio - 1`. Here the real
//! ranking feeds the real reader and the host recomputes both.
//!
//! # The three gates
//!
//! ```text
//! (a) the chain     — `index_topk` at ratio 4 over a paged index-key cache,
//!                     its selection fed straight to `pool_lse_selected` over
//!                     the compressed cache. Compared against a host that
//!                     scores `sum_h relu(q_h . k_c) * w_h` itself, bisects
//!                     with `index::bisect_select` (the pinned host twin of
//!                     the shader's forty halvings), and softmaxes over the
//!                     cells that selection names.
//! (b) the reduction — with a budget at or above the visible count the
//!                     ranking publishes the identity, so the selected reader
//!                     must land the DENSE reader's own `o` and `lse`. That
//!                     equality is what makes the fine branch safe to fire on
//!                     short sequences, and it is measured against
//!                     `pool_lse` on the card rather than argued.
//! (c) the fold      — `merge_lse` over (a dense causal branch, the selected
//!                     branch) closed by `attention.sink`, against a host
//!                     that runs ONE softmax over the union of the two key
//!                     sets with the per-head sink in the denominator alone.
//!                     That union is the reference oracle's own attention
//!                     (`oracle/step12_glue.py`: `sparse_attn(q,
//!                     concat(window, comp), attn_sink, widx + cidx)`), so
//!                     this gate is the cadence and not just the kernel.
//! ```
//!
//! Gate (c)'s first branch is `pool_lse` at ratio 1 — a dense causal read of
//! every cached row — standing in for the sliding window `prefill_lse`
//! serves in the model. It is the same LSE contract and the same merge; what
//! it does NOT exercise is the prefill plan, which `mla_on_device` and the
//! first-light files carry.
//!
//! # Gating
//!
//! As `pool_on_device`: `cfg`'d to Apple at compile time, and SKIPS at run
//! time when `device::present()` says no.
//!
//! ```text
//! cargo test -p engine-metal --test nsa_selected_on_device -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{KvPool, Tensor};
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, for `device_floor`'s reason.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------
// The geometry.
// ---------------------------------------------------------------------------

/// The compressed entry's width — the pooled reader's head.
const HEAD_DIM: usize = 16;
/// The indexer key's width. Narrower than the entry it ranks, exactly as
/// dsv4-flash's 128 is narrower than the 512 it ranks.
const KEY_DIM: usize = 8;
/// The indexer's head count.
const IX_HEADS: usize = 4;
/// The pooled reader's query heads.
const Q_HEADS: usize = 3;
const RATIO: usize = 4;
const PAGE_SIZE: usize = 16;
const PAGES_PER_REQ: usize = 6;
const REQUESTS: usize = 2;
const CELLS: usize = PAGE_SIZE * PAGES_PER_REQ * REQUESTS;

/// **SHUFFLED ON PURPOSE**, for `pool_on_device`'s reason: a kernel that took
/// the position for the cell reads a self-consistent plane and answers wrong.
/// Both the index cache and the compressed cache are addressed through it.
const PAGE_INDICES: [u32; PAGES_PER_REQ * REQUESTS] = [7, 2, 10, 0, 5, 9, 3, 11, 1, 8, 4, 6];

fn page_indptr() -> Vec<u32> {
    (0..=REQUESTS).map(|r| (r * PAGES_PER_REQ) as u32).collect()
}

fn paged_slot(req: usize, pos: usize) -> usize {
    let page = PAGE_INDICES[req * PAGES_PER_REQ + pos / PAGE_SIZE] as usize;
    page * PAGE_SIZE + pos % PAGE_SIZE
}

/// The cell the compressed row `c` of a `ratio`-pool lives in — the one piece
/// of arithmetic the ranking and the reader must agree on, written once here
/// so the host cannot drift from either.
fn boundary_cell(c: usize, ratio: usize) -> usize {
    (c + 1) * ratio - 1
}

// ---------------------------------------------------------------------------
// The fixture.
// ---------------------------------------------------------------------------

/// One case. Every plane is already through the element the device stores it
/// in, so the host and the card read the SAME numbers.
struct Fixture {
    /// `[rows]` — each query row's absolute position and owning request.
    positions: Vec<i32>,
    requests: Vec<i32>,
    /// `[rows, IX_HEADS * KEY_DIM]` bf16 — the indexer query.
    ix_q: Vec<f32>,
    /// `[rows, IX_HEADS]` bf16 — the learned per-head combine weights.
    ix_w: Vec<f32>,
    /// `[CELLS, KEY_DIM]` bf16 — the index key cache, written at boundary
    /// cells alone, exactly as `pool_kv_append` writes the pooled key.
    keys: Vec<f32>,
    /// `[rows, Q_HEADS * HEAD_DIM]` bf16 — the pooled reader's query.
    q: Vec<f32>,
    /// `[CELLS, HEAD_DIM]` bf16 — the compressed entry cache.
    entries: Vec<f32>,
    /// `[Q_HEADS]` bf16 — the per-head attention sink. `attn_sink_rescale`
    /// reads the sink plane at the ACTIVATION's element, not f32: it is the
    /// `const device T*` beside `o`, and a test that staged f32 here would
    /// hand the shader two halves of two numbers.
    sink: Vec<f32>,
}

impl Fixture {
    fn new(seed: u64, positions: Vec<i32>, requests: Vec<i32>) -> Self {
        let rows = positions.len();
        let mut rng = Lcg(seed);
        let mut keys = vec![0.0f32; CELLS * KEY_DIM];
        let mut entries = vec![0.0f32; CELLS * HEAD_DIM];
        // Only the BOUNDARY cells hold anything. A ranking that scanned every
        // position instead of striding by the ratio would score zero rows and
        // select a set the host's own stride never names.
        for req in 0..REQUESTS {
            for c in 0..(PAGE_SIZE * PAGES_PER_REQ) / RATIO {
                let slot = paged_slot(req, boundary_cell(c, RATIO));
                for d in 0..KEY_DIM {
                    keys[slot * KEY_DIM + d] = rng.bf16();
                }
                for d in 0..HEAD_DIM {
                    entries[slot * HEAD_DIM + d] = rng.bf16();
                }
            }
        }
        Self {
            ix_q: rng.bf16_plane(rows * IX_HEADS * KEY_DIM),
            ix_w: rng.bf16_plane(rows * IX_HEADS),
            q: rng.bf16_plane(rows * Q_HEADS * HEAD_DIM),
            positions,
            requests,
            keys,
            entries,
            sink: (0..Q_HEADS)
                .map(|_| f32_of(bf16_bits(rng.next_f32() * 4.0)))
                .collect(),
        }
    }

    fn rows(&self) -> usize {
        self.positions.len()
    }

    fn visible(&self, t: usize) -> usize {
        ((self.positions[t] + 1) as usize) / RATIO
    }

    /// `I(t, c) = sum_h relu(q_h . k_c) * w_h` over the visible compressed
    /// rows — `index.cuh`'s statement, in f32, at this family's stride.
    fn scores(&self, t: usize) -> Vec<f32> {
        let req = self.requests[t] as usize;
        (0..self.visible(t))
            .map(|c| {
                let slot = paged_slot(req, boundary_cell(c, RATIO));
                (0..IX_HEADS)
                    .map(|h| {
                        let dot: f32 = (0..KEY_DIM)
                            .map(|d| {
                                self.ix_q[(t * IX_HEADS + h) * KEY_DIM + d]
                                    * self.keys[slot * KEY_DIM + d]
                            })
                            .sum();
                        dot.max(0.0) * self.ix_w[t * IX_HEADS + h]
                    })
                    .sum()
            })
            .collect()
    }

    /// The selection the shader must publish, through the pinned host twin of
    /// its own bisection.
    fn selection(&self, t: usize, top_k: usize) -> Vec<i32> {
        kernels_metal::attn::index::bisect_select(&self.scores(t), top_k)
    }

    /// The softmax the pooled reader runs over a stated set of compressed
    /// rows: `o` and the BASE-2 log-sum-exp the cascade merge folds.
    fn read(&self, t: usize, head: usize, rows: &[usize], scale: f32) -> (Vec<f32>, f32) {
        let req = self.requests[t] as usize;
        let logits: Vec<f32> = rows
            .iter()
            .map(|c| {
                let slot = paged_slot(req, boundary_cell(*c, RATIO));
                let dot: f32 = (0..HEAD_DIM)
                    .map(|d| {
                        self.q[(t * Q_HEADS + head) * HEAD_DIM + d]
                            * self.entries[slot * HEAD_DIM + d]
                    })
                    .sum();
                dot * scale
            })
            .collect();
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, l| m.max(*l));
        if !max.is_finite() {
            return (vec![0.0; HEAD_DIM], f32::NEG_INFINITY);
        }
        let mut z = 0.0f32;
        let mut acc = vec![0.0f32; HEAD_DIM];
        for (l, c) in logits.iter().zip(rows) {
            let w = (l - max).exp();
            z += w;
            let slot = paged_slot(req, boundary_cell(*c, RATIO));
            for d in 0..HEAD_DIM {
                acc[d] += w * self.entries[slot * HEAD_DIM + d];
            }
        }
        if z <= 0.0 {
            return (vec![0.0; HEAD_DIM], f32::NEG_INFINITY);
        }
        for a in &mut acc {
            *a /= z;
        }
        (acc, (z.ln() + max) * std::f32::consts::LOG2_E)
    }
}

// ---------------------------------------------------------------------------
// The device side.
// ---------------------------------------------------------------------------

/// Everything staged once, so a gate can fire several entries against one
/// residency and read whichever answers it asked for.
struct Staged {
    device: Context,
    pipelines: Pipelines,
    handles: Handles,
    keys: Buffer,
    entries: Buffer,
    ix_q: Buffer,
    ix_w: Buffer,
    q: Buffer,
    sink: Buffer,
    positions: Buffer,
    requests: Buffer,
    indices: Buffer,
    indptr: Buffer,
    rows: u32,
}

impl Staged {
    fn new(fx: &Fixture) -> Self {
        let device = Context::bind().expect("the device binds");
        Self {
            keys: staged(&device, &encode_bf16(&fx.keys)),
            entries: staged(&device, &encode_bf16(&fx.entries)),
            ix_q: staged(&device, &encode_bf16(&fx.ix_q)),
            ix_w: staged(&device, &encode_bf16(&fx.ix_w)),
            q: staged(&device, &encode_bf16(&fx.q)),
            sink: staged(&device, &encode_bf16(&fx.sink)),
            positions: staged(&device, &encode_i32(&fx.positions)),
            requests: staged(&device, &encode_i32(&fx.requests)),
            indices: staged(&device, &encode_u32(&PAGE_INDICES)),
            indptr: staged(&device, &encode_u32(&page_indptr())),
            rows: fx.rows() as u32,
            pipelines: Pipelines::new(),
            handles: Handles::new(),
            device,
        }
    }

    fn whole(&self, buffer: &Buffer, what: &str) -> u32 {
        self.handles
            .bind(buffer, 0, buffer.bytes())
            .unwrap_or_else(|fault| panic!("{what} binds: {fault}"))
    }

    /// A paged cache over `plane` at `width`, sharing the one page map.
    fn pool(&self, plane: &Buffer, width: usize, what: &str) -> KvPool {
        let keys = Tensor::new(self.whole(plane, what), CELLS as u32, width as u32, Dtype::Bf16);
        KvPool {
            keys,
            values: keys,
            page_indices: Tensor::new(
                self.whole(&self.indices, "the page map"),
                PAGE_INDICES.len() as u32,
                1,
                Dtype::U32,
            ),
            page_indptr: Tensor::new(
                self.whole(&self.indptr, "the page spans"),
                (REQUESTS + 1) as u32,
                1,
                Dtype::U32,
            ),
            page_size: PAGE_SIZE as i32,
            seq_stride: width as u64,
            head_stride: width as u64,
        }
    }

    fn i32_rows(&self, buffer: &Buffer, what: &str) -> Tensor {
        Tensor::new(self.whole(buffer, what), self.rows, 1, Dtype::I32)
    }

    fn bf16(&self, buffer: &Buffer, width: usize, what: &str) -> Tensor {
        Tensor::new(self.whole(buffer, what), self.rows, width as u32, Dtype::Bf16)
    }
}

/// Fire the REAL ranking and read its selection back. `scores` is the working
/// slab the shader bisects over — `crate::scratch`'s index role on the shell
/// side, staged here because this file is the shell.
fn rank(st: &Staged, top_k: usize) -> Vec<i32> {
    let rows = st.rows as usize;
    let stride = PAGE_SIZE * PAGES_PER_REQ;
    let scores = Buffer::zeroed(&st.device, (rows * stride * 4) as u64).expect("the slab reserves");
    let out = Buffer::zeroed(&st.device, (rows * top_k * 4) as u64).expect("the selection reserves");
    let frame = st.device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&st.device, &frame, &st.pipelines, &st.handles);
        kernels_metal::attn::index::topk(
            &sink,
            st.bf16(&st.ix_q, IX_HEADS * KEY_DIM, "the indexer query"),
            st.bf16(&st.ix_w, IX_HEADS, "the indexer head weights"),
            &st.pool(&st.keys, KEY_DIM, "the index key cache"),
            st.i32_rows(&st.positions, "positions"),
            st.i32_rows(&st.requests, "request_of_token"),
            Tensor::new(
                st.whole(&scores, "the score slab"),
                st.rows,
                stride as u32,
                Dtype::F32,
            ),
            IX_HEADS as u32,
            KEY_DIM as u32,
            top_k as u32,
            RATIO as u32,
            Tensor::new(
                st.whole(&out, "the selection"),
                st.rows,
                top_k as u32,
                Dtype::I32,
            ),
        )
        .expect("the ranking encodes");
    }
    frame.commit().expect("the ranking completes");
    decode_i32(&read_back(&out, rows * top_k * 4))
}

/// Fire the selected pooled reader over a selection buffer already on the
/// card, and read `(o, lse)` back.
fn read_selected(st: &Staged, selection: &Buffer, top_k: usize, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let rows = st.rows as usize;
    let o = Buffer::zeroed(&st.device, (rows * Q_HEADS * HEAD_DIM * 2) as u64).expect("o reserves");
    let lse = Buffer::zeroed(&st.device, (rows * Q_HEADS * 4) as u64).expect("lse reserves");
    let frame = st.device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&st.device, &frame, &st.pipelines, &st.handles);
        kernels_metal::attn::pool::attention_lse_selected(
            &sink,
            st.bf16(&st.q, Q_HEADS * HEAD_DIM, "the reader's query"),
            st.i32_rows(&st.positions, "positions"),
            st.i32_rows(&st.requests, "request_of_token"),
            Tensor::new(
                st.whole(selection, "the selection"),
                st.rows,
                top_k as u32,
                Dtype::I32,
            ),
            &st.pool(&st.entries, HEAD_DIM, "the compressed cache"),
            RATIO as u32,
            top_k as u32,
            Q_HEADS as u32,
            HEAD_DIM as u32,
            scale,
            st.bf16(&o, Q_HEADS * HEAD_DIM, "o"),
            Tensor::new(st.whole(&lse, "lse"), st.rows, Q_HEADS as u32, Dtype::F32),
        )
        .expect("the selected reader encodes");
    }
    frame.commit().expect("the selected reader completes");
    (
        decode_bf16(&read_back(&o, rows * Q_HEADS * HEAD_DIM * 2)),
        decode_f32(&read_back(&lse, rows * Q_HEADS * 4)),
    )
}

/// The DENSE pooled reader at `ratio`, for the two gates that hold the
/// selected one against it.
fn read_dense(st: &Staged, ratio: usize, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let rows = st.rows as usize;
    let o = Buffer::zeroed(&st.device, (rows * Q_HEADS * HEAD_DIM * 2) as u64).expect("o reserves");
    let lse = Buffer::zeroed(&st.device, (rows * Q_HEADS * 4) as u64).expect("lse reserves");
    let frame = st.device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&st.device, &frame, &st.pipelines, &st.handles);
        kernels_metal::attn::pool::attention_lse(
            &sink,
            st.bf16(&st.q, Q_HEADS * HEAD_DIM, "the reader's query"),
            st.i32_rows(&st.positions, "positions"),
            st.i32_rows(&st.requests, "request_of_token"),
            &st.pool(&st.entries, HEAD_DIM, "the compressed cache"),
            ratio as u32,
            Q_HEADS as u32,
            HEAD_DIM as u32,
            scale,
            st.bf16(&o, Q_HEADS * HEAD_DIM, "o"),
            Tensor::new(st.whole(&lse, "lse"), st.rows, Q_HEADS as u32, Dtype::F32),
        )
        .expect("the dense reader encodes");
    }
    frame.commit().expect("the dense reader completes");
    (
        decode_bf16(&read_back(&o, rows * Q_HEADS * HEAD_DIM * 2)),
        decode_f32(&read_back(&lse, rows * Q_HEADS * 4)),
    )
}

// ---------------------------------------------------------------------------
// (a) THE CHAIN
// ---------------------------------------------------------------------------

/// **THE RANKING'S IDS AND THE READER'S CELLS ARE ONE SPACE.**
///
/// Nothing here types a selection. The card ranks, the card reads what it
/// ranked, and the host recomputes both ends from the same planes. A ranking
/// that scanned positions instead of striding by the ratio would score cells
/// this fixture left at zero; a reader that took the id for the cell would
/// read the wrong quarter of the cache. Either is a different `o`.
#[test]
fn the_indexer_ranks_and_the_reader_walks_what_it_ranked() {
    let _one = serialized();
    if !device::present() {
        println!("SKIP the chain: this machine publishes no Metal device");
        return;
    }
    // Positions chosen so the visible compressed count straddles the budget:
    // rows 0-1 see fewer rows than the budget (the identity arm), rows 2-4
    // see more (the bisection arm).
    let fx = Fixture::new(
        0x51ec_7ed0,
        vec![7, 15, 47, 83, 95],
        vec![0, 0, 1, 1, 0],
    );
    let top_k = 6;
    let scale = 0.25;
    let st = Staged::new(&fx);

    let got = rank(&st, top_k);
    let mut selected_rows = 0usize;
    for t in 0..fx.rows() {
        let want = fx.selection(t, top_k);
        assert_eq!(
            &got[t * top_k..(t + 1) * top_k],
            &want[..],
            "row {t} (position {}, {} visible compressed rows) ranked differently \
             than the host's forty halvings",
            fx.positions[t],
            fx.visible(t)
        );
        if fx.visible(t) > top_k {
            selected_rows += 1;
        }
    }
    assert!(
        selected_rows >= 2,
        "no row exceeded its budget, so the bisection arm was never asked"
    );

    // The selection the CARD published, handed straight back to the card.
    let selection = staged(&st.device, &encode_i32(&got));
    let (o, lse) = read_selected(&st, &selection, top_k, scale);

    let mut want_o = vec![0.0f32; o.len()];
    let mut worst_lse = 0.0f32;
    for t in 0..fx.rows() {
        let chosen: Vec<usize> = got[t * top_k..(t + 1) * top_k]
            .iter()
            .filter(|c| **c >= 0 && (**c as usize) < fx.visible(t))
            .map(|c| *c as usize)
            .collect();
        for h in 0..Q_HEADS {
            let (row, want_lse) = fx.read(t, h, &chosen, scale);
            let at = (t * Q_HEADS + h) * HEAD_DIM;
            want_o[at..at + HEAD_DIM].copy_from_slice(&row);
            worst_lse = worst_lse.max((lse[t * Q_HEADS + h] - want_lse).abs());
        }
    }
    let worst_o = worst_in_quanta(&o, &want_o);
    println!(
        "(a) chain: {} rows, budget {top_k}, {selected_rows} rows past it — worst o \
         {worst_o:.3} bf16 quanta, worst lse {worst_lse:.2e}",
        fx.rows()
    );
    assert!(worst_o < 1.5, "the selected read drifted {worst_o:.3} quanta");
    assert!(worst_lse < 2e-3, "the selected lse drifted {worst_lse:.2e}");
}

// ---------------------------------------------------------------------------
// (b) THE REDUCTION
// ---------------------------------------------------------------------------

/// **A BUDGET NOTHING EXCEEDS IS THE DENSE READER.**
///
/// `index_topk_paged`'s `nkeys <= topk` arm publishes `0..nkeys-1` padded with
/// `-1`, so the selected reader must walk the same cells in the same order the
/// dense one walks and land the same `o` and `lse` — not close, the same
/// arithmetic. This is what lets the fine branch fire on a five-layer
/// miniature whose sequences never reach the trained `index_topk`.
#[test]
fn a_budget_nothing_exceeds_lands_the_dense_readers_own_numbers() {
    let _one = serialized();
    if !device::present() {
        println!("SKIP the reduction: this machine publishes no Metal device");
        return;
    }
    let fx = Fixture::new(0x0dd_ba11, vec![11, 23, 35, 43], vec![0, 1, 0, 1]);
    // Every row sees at most 11 compressed rows; the budget is above that.
    let top_k = 16;
    let scale = 0.125;
    let st = Staged::new(&fx);

    let got = rank(&st, top_k);
    for t in 0..fx.rows() {
        let visible = fx.visible(t);
        assert!(visible <= top_k, "row {t} was meant to fit its budget");
        for (n, id) in got[t * top_k..(t + 1) * top_k].iter().enumerate() {
            let want = if n < visible { n as i32 } else { -1 };
            assert_eq!(*id, want, "row {t} slot {n} is not the identity selection");
        }
    }

    let selection = staged(&st.device, &encode_i32(&got));
    let (o_sel, lse_sel) = read_selected(&st, &selection, top_k, scale);
    let (o_dense, lse_dense) = read_dense(&st, RATIO, scale);
    assert_eq!(o_sel, o_dense, "the selected reader did not land the dense `o`");
    assert_eq!(
        lse_sel, lse_dense,
        "the selected reader did not land the dense `lse`"
    );
    println!(
        "(b) reduction: {} rows x {Q_HEADS} heads identical to `pool_lse`",
        fx.rows()
    );
}

// ---------------------------------------------------------------------------
// (c) THE FOLD
// ---------------------------------------------------------------------------

/// **THE CADENCE, NOT JUST THE KERNEL.**
///
/// The reference oracle's attention is ONE softmax over the union of a
/// sliding window and the visible compressed rows, with the per-head
/// `attn_sink` added to the DENOMINATOR alone. pie computes that as two LSE
/// branches folded by `merge_lse` and closed by `attention.sink`. This gate
/// puts the SELECTED branch in that fold and holds the result against the
/// single-softmax reference, so a selected branch whose log-sum-exp was in
/// the wrong base — or whose `-inf` for an empty selection poisoned the merge
/// — fails here even though gate (a) passed.
///
/// The first branch is `pool_lse` at ratio 1 (every cached row, causally),
/// standing in for the window `prefill_lse` serves in the model: the same
/// base-2 LSE contract, without a prefill plan this file would otherwise have
/// to carve.
#[test]
fn the_selected_branch_folds_into_the_reference_s_one_softmax() {
    let _one = serialized();
    if !device::present() {
        println!("SKIP the fold: this machine publishes no Metal device");
        return;
    }
    let fx = Fixture::new(0xf01d_ed, vec![31, 59, 87], vec![0, 1, 1]);
    let top_k = 5;
    let scale = 0.125;
    let st = Staged::new(&fx);

    let selection = staged(&st.device, &encode_i32(&rank(&st, top_k)));
    let chosen: Vec<Vec<usize>> = {
        let ids = decode_i32(&read_back(&selection, fx.rows() * top_k * 4));
        (0..fx.rows())
            .map(|t| {
                ids[t * top_k..(t + 1) * top_k]
                    .iter()
                    .filter(|c| **c >= 0 && (**c as usize) < fx.visible(t))
                    .map(|c| *c as usize)
                    .collect()
            })
            .collect()
    };

    let rows = fx.rows();
    let o = Buffer::zeroed(&st.device, (rows * Q_HEADS * HEAD_DIM * 2) as u64).expect("o reserves");
    let lse = Buffer::zeroed(&st.device, (rows * Q_HEADS * 4) as u64).expect("lse reserves");
    let o1 = Buffer::zeroed(&st.device, (rows * Q_HEADS * HEAD_DIM * 2) as u64).expect("o1");
    let lse1 = Buffer::zeroed(&st.device, (rows * Q_HEADS * 4) as u64).expect("lse1");
    let o2 = Buffer::zeroed(&st.device, (rows * Q_HEADS * HEAD_DIM * 2) as u64).expect("o2");
    let lse2 = Buffer::zeroed(&st.device, (rows * Q_HEADS * 4) as u64).expect("lse2");

    let plane = |b: &Buffer, what: &str| st.bf16(b, Q_HEADS * HEAD_DIM, what);
    let column =
        |b: &Buffer, what: &str| Tensor::new(st.whole(b, what), st.rows, Q_HEADS as u32, Dtype::F32);

    let frame = st.device.frame().expect("a command buffer opens");
    {
        let enc = Sink::new(&st.device, &frame, &st.pipelines, &st.handles);
        let q = plane(&st.q, "the reader's query");
        let positions = st.i32_rows(&st.positions, "positions");
        let requests = st.i32_rows(&st.requests, "request_of_token");
        let entries = st.pool(&st.entries, HEAD_DIM, "the compressed cache");
        // Branch one: the dense causal read, standing in for the window.
        kernels_metal::attn::pool::attention_lse(
            &enc, q, positions, requests, &entries, 1, Q_HEADS as u32, HEAD_DIM as u32,
            scale, plane(&o1, "o1"), column(&lse1, "lse1"),
        )
        .expect("the dense branch encodes");
        // Branch two: the selected compressed read.
        kernels_metal::attn::pool::attention_lse_selected(
            &enc,
            q,
            positions,
            requests,
            Tensor::new(
                st.whole(&selection, "the selection"),
                st.rows,
                top_k as u32,
                Dtype::I32,
            ),
            &entries,
            RATIO as u32,
            top_k as u32,
            Q_HEADS as u32,
            HEAD_DIM as u32,
            scale,
            plane(&o2, "o2"),
            column(&lse2, "lse2"),
        )
        .expect("the selected branch encodes");
        kernels_metal::attn::merge_lse(
            &enc,
            plane(&o1, "o1 read"),
            column(&lse1, "lse1 read"),
            plane(&o2, "o2 read"),
            column(&lse2, "lse2 read"),
            Q_HEADS as u32,
            HEAD_DIM as u32,
            plane(&o, "the merged o"),
            column(&lse, "the merged lse"),
        )
        .expect("the merge encodes");
        kernels_metal::attn::sink(
            &enc,
            plane(&o, "the merged o"),
            column(&lse, "the merged lse"),
            Tensor::new(st.whole(&st.sink, "the sink"), Q_HEADS as u32, 1, Dtype::Bf16),
            HEAD_DIM as u32,
        )
        .expect("the sink encodes");
    }
    frame.commit().expect("the fold completes");
    let got = decode_bf16(&read_back(&o, rows * Q_HEADS * HEAD_DIM * 2));

    // The host: ONE softmax over the union, sink in the denominator alone.
    let mut want = vec![0.0f32; got.len()];
    for t in 0..rows {
        let dense: Vec<usize> = (0..(fx.positions[t] + 1) as usize).collect();
        for h in 0..Q_HEADS {
            let req = fx.requests[t] as usize;
            let mut logits = Vec::new();
            let mut vals = Vec::new();
            let push = |cell: usize, logits: &mut Vec<f32>, vals: &mut Vec<usize>| {
                let slot = paged_slot(req, cell);
                let dot: f32 = (0..HEAD_DIM)
                    .map(|d| {
                        fx.q[(t * Q_HEADS + h) * HEAD_DIM + d] * fx.entries[slot * HEAD_DIM + d]
                    })
                    .sum();
                logits.push(dot * scale);
                vals.push(slot);
            };
            for c in &dense {
                push(boundary_cell(*c, 1), &mut logits, &mut vals);
            }
            for c in &chosen[t] {
                push(boundary_cell(*c, RATIO), &mut logits, &mut vals);
            }
            let max = logits.iter().fold(fx.sink[h], |m, l| m.max(*l));
            let mut z = (fx.sink[h] - max).exp();
            let mut acc = vec![0.0f32; HEAD_DIM];
            for (l, slot) in logits.iter().zip(&vals) {
                let w = (l - max).exp();
                z += w;
                for d in 0..HEAD_DIM {
                    acc[d] += w * fx.entries[slot * HEAD_DIM + d];
                }
            }
            for d in 0..HEAD_DIM {
                want[(t * Q_HEADS + h) * HEAD_DIM + d] = acc[d] / z;
            }
        }
    }
    let worst = worst_in_quanta(&got, &want);
    assert!(
        want.iter().any(|w| w.abs() > 1e-3),
        "the reference folded to nothing, so this gate proves nothing"
    );
    println!(
        "(c) fold: merge_lse(dense, selected) + sink over {rows} rows x {Q_HEADS} heads \
         — worst {worst:.3} bf16 quanta against one softmax over the union"
    );
    assert!(
        worst < 1.5,
        "the folded branches drifted {worst:.3} quanta from the single softmax"
    );
}

// ---------------------------------------------------------------------------
// Host staging — `pool_on_device`'s helpers, for its reasons.
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        (x - 0.5) * 0.5
    }

    fn bf16(&mut self) -> f32 {
        f32_of(bf16_bits(self.next_f32()))
    }

    fn bf16_plane(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.bf16()).collect()
    }
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn f32_of(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32_of(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The bf16 quantum at `v`: eight significant bits below the binade.
fn quantum(v: f32) -> f32 {
    if v == 0.0 {
        return f32::MIN_POSITIVE;
    }
    v.abs().log2().floor().exp2() / 128.0
}

/// The worst absolute miss, in bf16 quanta OF THE PLANE'S SCALE —
/// `pool_on_device`'s metric, for its reason: against each element's own
/// quantum, an output that happened to cancel to near zero would divide a
/// rounding-sized miss by a denormal and report a number about nothing.
fn worst_in_quanta(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    worst / quantum(scale)
}

fn staged(device: &Context, bytes: &[u8]) -> Buffer {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("the reservation lands");
    buffer.write(0, bytes).expect("the bytes land");
    buffer
}

fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut got = vec![0u8; bytes];
    buffer.read(0, &mut got).expect("the answer reads back");
    got
}
