//! The dsv4 compressor's gated softmax pool, on a real Apple GPU.
//!
//! **WHAT THIS FILE IS FOR.** `attention.pool_gather` was the last fatal
//! refusal on the dsv4 rows, and it was fatal for a seam rather than for a
//! kernel: the shader shipped with the rest of `attn/pool.metal` and nothing
//! could reach it, because the three planes it reads beside the cache
//! (`state_kv`, `state_score`, `ape`) were named by no IR op. The seam is
//! crossed — `crate::scratch` reserves the two state slabs at the source
//! paging's cell ceiling and the dispatch arm binds them, and `ape` took an
//! operand of its own — so the arithmetic is reachable and this file measures
//! it.
//!
//! **AND THE STATE HAS A WRITER NOW.** Gates (a)-(d) hand-build the state
//! plane and ask whether the gather reads it correctly; none of them can ask
//! whether anything in the stack could put numbers there, because for as long
//! as the compressor's four planes were interned nothing could. Gate (e) is
//! the round trip: `attention.pool_state_write` scatters the `wkv`/`wgate`
//! projections into the cells `write_page`/`write_offset` name, and the
//! gather pools them back out through the page table from the other end.
//!
//! # The three things a faithful port of this kernel can still get wrong
//!
//! ```text
//! (a) the WINDOW      pos = bpos + i - (coff·ratio - 1), i in [0, coff·ratio)
//!                     — an off-by-one pools the wrong block and still runs
//! (b) the COLUMN      col = (i >= ratio ? head_dim : 0) + d — the ratio-4
//!                     compressor emits a k/v PAIR per cell, and the second
//!                     half of the window reads the second half of the row.
//!                     A port that dropped the `coff` fan would read one half
//!                     twice and produce a plausible, wrong entry
//! (c) the ADDRESS     the state is indexed by the source pool's PAGED SLOT,
//!                     not by the position — so the page table has to be
//!                     walked, and this file's page map is deliberately
//!                     shuffled so a kernel that skipped it fails
//! ```
//!
//! Each gate runs the real entry against a host fp32 reference that walks the
//! same window, the same columns and the same page map. The band is a bf16
//! quantum, because the entry lands bf16 and everything above it is f32.
//!
//! **AND THE MARKS THEMSELVES HAVE A GATE NOW.** Gates (a)-(e) all take the
//! boundary tables as GIVEN: the fixture types `bpos` and asks what the pool
//! does at those marks. Gate (f) fires the mark kernels and measures the two
//! DIFFERENT positions one pooled entry has — the CELL it is cached at
//! (`(c+1)·ratio - 1`) and the COMPRESSED ROW it is roped at (`c·ratio`, the
//! reference's `arange(0, cutoff, ratio)`). They are `ratio - 1` apart, and
//! for as long as the mark published only the cell the model text roped at
//! the wrong one.
//!
//! # The six gates
//!
//! ```text
//! (a) ratio 4    — coff 2, the overlapping 2·ratio window, ape ON
//! (b) ratio 128  — coff 1, the plain window, ape OFF (the CUDA nullptr path)
//! (c) has_ape    — an ape plane of zeros is the absent plane, exactly
//! (d) has_ape    — an ape plane of anything else is NOT, so the flag is read
//! (e) round trip — `pool_state_write` writing into zeroed slabs, and the
//!                  gather reading its cells back through the shuffled map
//! (f) the marks  — `pool_boundary_{prefill,decode}`'s three columns, the
//!                  rope one against `arange(0, cutoff, ratio)`
//! ```
//!
//! Gates (c) and (d) are one claim in two directions: `has_ape` is the Metal
//! spelling of the CUDA `ape != nullptr`, and a shell that bound the seat but
//! never stated the flag — or stated it and never read the plane — would pass
//! one of them and fail the other.
//!
//! # Gating
//!
//! As `device_floor`, `mla_on_device` and `hc_on_device`: `cfg`'d to Apple at
//! compile time, and SKIPS at run time when `device::present()` says no.
//!
//! ```text
//! cargo test -p engine-metal --test pool_on_device -- --nocapture
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

fn device_or_skip(what: &str) -> Option<Context> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    Some(Context::bind().expect("the device binds"))
}

// ---------------------------------------------------------------------------
// The geometry. Small enough to hold in a test, wide enough that every organ
// of the addressing is exercised: two requests, eight pages each, a shuffled
// page map, and a context long enough for a ratio-128 window to close twice.
// ---------------------------------------------------------------------------

const HEAD_DIM: usize = 8;
const PAGE_SIZE: usize = 32;
const PAGES_PER_REQ: usize = 8;
const REQUESTS: usize = 2;
/// Every cell the pool holds, which is what a state slab has a row for.
const CELLS: usize = PAGE_SIZE * PAGES_PER_REQ * REQUESTS;

/// **THE PAGE MAP IS SHUFFLED ON PURPOSE.** A kernel that took the position
/// for the cell — `slot = pos` instead of `page_indices[...]·page_size +
/// pos % page_size` — would read a self-consistent plane and produce a wrong
/// answer that no shape check can see. With this map it reads someone else's
/// block and the band catches it.
const PAGE_INDICES: [u32; PAGES_PER_REQ * REQUESTS] =
    [3, 11, 7, 0, 5, 2, 9, 14, 8, 6, 1, 10, 12, 13, 4, 15];

fn page_indptr() -> Vec<u32> {
    (0..=REQUESTS).map(|r| (r * PAGES_PER_REQ) as u32).collect()
}

/// The cell a `(request, position)` pair addresses — the host twin of the
/// shader's `pool_paged_slot`.
fn paged_slot(req: usize, pos: usize) -> usize {
    let page = PAGE_INDICES[req * PAGES_PER_REQ + pos / PAGE_SIZE] as usize;
    page * PAGE_SIZE + pos % PAGE_SIZE
}

/// The compressor's window fanout: `2` at ratio 4, `1` elsewhere.
fn coff(ratio: usize) -> usize {
    if ratio == 4 { 2 } else { 1 }
}

// ---------------------------------------------------------------------------
// The fixture and the reference.
// ---------------------------------------------------------------------------

/// One case: the boundary tables and the three planes, already through the
/// element the device stores them in, so the reference and the device read
/// the SAME numbers and the only difference either can show is arithmetic.
#[derive(Clone)]
struct Fixture {
    ratio: usize,
    width: usize,
    /// `[rows]` — the closing position of each entry, `-1` for a masked row.
    bpos: Vec<i32>,
    /// `[rows]` — the owning request of each entry.
    breq: Vec<i32>,
    /// `[CELLS, width]` bf16 — the rolling kv window.
    state_kv: Vec<f32>,
    /// `[CELLS, width]` bf16 — the rolling gate logits.
    state_score: Vec<f32>,
    /// `[ratio, width]` f32 — the intra-block absolute-position plane.
    ape: Vec<f32>,
}

impl Fixture {
    fn new(seed: u64, ratio: usize, bpos: Vec<i32>, breq: Vec<i32>) -> Self {
        let width = coff(ratio) * HEAD_DIM;
        let mut rng = Lcg(seed);
        Self {
            ratio,
            width,
            bpos,
            breq,
            state_kv: rng.bf16_plane(CELLS * width),
            state_score: rng.bf16_plane(CELLS * width),
            // Wider than the scores, so the position plane is never a
            // rounding-sized nudge the band could swallow.
            ape: (0..ratio * width).map(|_| rng.next_f32() * 6.0).collect(),
        }
    }

    fn rows(&self) -> usize {
        self.bpos.len()
    }

    /// The host's own gated pool, walked exactly as `pool_gather_paged`
    /// walks it: one pass for the max, one for the weighted sum, both in f32
    /// and both in the shader's order.
    fn reference(&self, has_ape: bool) -> Vec<f32> {
        let window = coff(self.ratio) * self.ratio;
        let mut out = vec![0.0f32; self.rows() * HEAD_DIM];
        for c in 0..self.rows() {
            let bpos = self.bpos[c];
            if bpos < 0 {
                continue;
            }
            let req = self.breq[c] as usize;
            for d in 0..HEAD_DIM {
                let mut cells = Vec::new();
                for i in 0..window {
                    let pos = bpos + i as i32 - (window as i32 - 1);
                    if pos < 0 {
                        continue;
                    }
                    let pos = pos as usize;
                    let col = if i >= self.ratio { HEAD_DIM } else { 0 } + d;
                    let slot = paged_slot(req, pos);
                    let mut sc = self.state_score[slot * self.width + col];
                    if has_ape {
                        sc += self.ape[(pos % self.ratio) * self.width + col];
                    }
                    cells.push((sc, self.state_kv[slot * self.width + col]));
                }
                let max_s = cells.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
                if !max_s.is_finite() {
                    continue;
                }
                let (mut sum_e, mut acc) = (0.0f32, 0.0f32);
                for (s, v) in &cells {
                    let e = (s - max_s).exp();
                    sum_e += e;
                    acc += e * v;
                }
                out[c * HEAD_DIM + d] = if sum_e > 0.0 { acc / sum_e } else { 0.0 };
            }
        }
        out
    }
}

/// Fire the real entry over a fixture and read the pooled entries back.
fn fire(fx: &Fixture, has_ape: bool) -> Vec<f32> {
    let device = Context::bind().expect("the device binds");
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let state_kv = staged(&device, &encode_bf16(&fx.state_kv));
    let state_score = staged(&device, &encode_bf16(&fx.state_score));
    let ape = staged(&device, &encode_f32(&fx.ape));
    let bpos = staged(&device, &encode_i32(&fx.bpos));
    let breq = staged(&device, &encode_i32(&fx.breq));
    let indices = staged(&device, &encode_u32(&PAGE_INDICES));
    let indptr = staged(&device, &encode_u32(&page_indptr()));
    // The gather never reads the cache's storage — only its page tables — but
    // a `KvPool` states it, so it is a real (unread) reservation and not a
    // handle to nothing.
    let pages = Buffer::zeroed(&device, (CELLS * HEAD_DIM * 2) as u64).expect("the pool reserves");
    let out =
        Buffer::zeroed(&device, (fx.rows() * HEAD_DIM * 2) as u64).expect("the entries reserve");

    let cells = CELLS as u32;
    let width = fx.width as u32;
    let rows = fx.rows() as u32;
    let pool = KvPool {
        keys: Tensor::new(
            bind_whole(&handles, &pages, "the pool keys"),
            cells,
            HEAD_DIM as u32,
            Dtype::Bf16,
        ),
        values: Tensor::new(
            bind_whole(&handles, &pages, "the pool values"),
            cells,
            HEAD_DIM as u32,
            Dtype::Bf16,
        ),
        page_indices: Tensor::new(
            bind_whole(&handles, &indices, "the page map"),
            PAGE_INDICES.len() as u32,
            1,
            Dtype::U32,
        ),
        page_indptr: Tensor::new(
            bind_whole(&handles, &indptr, "the page spans"),
            (REQUESTS + 1) as u32,
            1,
            Dtype::U32,
        ),
        page_size: PAGE_SIZE as i32,
        seq_stride: HEAD_DIM as u64,
        head_stride: HEAD_DIM as u64,
    };

    let state_kv = Tensor::new(
        bind_whole(&handles, &state_kv, "state_kv"),
        cells,
        width,
        Dtype::Bf16,
    );
    let state_score = Tensor::new(
        bind_whole(&handles, &state_score, "state_score"),
        cells,
        width,
        Dtype::Bf16,
    );
    let ape = has_ape.then(|| {
        Tensor::new(
            bind_whole(&handles, &ape, "the position plane"),
            fx.ratio as u32,
            width,
            Dtype::F32,
        )
    });

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::pool::gather(
            &sink,
            Tensor::new(bind_whole(&handles, &bpos, "boundary_pos"), rows, 1, Dtype::I32),
            Tensor::new(bind_whole(&handles, &breq, "boundary_req"), rows, 1, Dtype::I32),
            &pool,
            HEAD_DIM as u32,
            fx.ratio as u32,
            state_kv,
            state_score,
            ape,
            Tensor::new(
                bind_whole(&handles, &out, "the pooled entries"),
                rows,
                HEAD_DIM as u32,
                Dtype::Bf16,
            ),
        )
        .expect("the gather encodes");
    }
    frame.commit().expect("the gather completes");
    decode_bf16(&read_back(&out, fx.rows() * HEAD_DIM * 2))
}

/// The worst absolute miss, in bf16 quanta OF THE PLANE'S SCALE.
///
/// Against each element's own quantum instead, an entry that happened to
/// cancel to near zero would divide a rounding-sized miss by a denormal and
/// report a number about nothing. The plane's largest reference value is the
/// honest denominator: bf16 rounds an element of that magnitude by at most
/// half a quantum and every smaller element by less, so this is the tighter
/// claim as well as the stable one.
fn worst_in_quanta(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    worst / quantum(scale)
}

// ---------------------------------------------------------------------------
// Gate (a): the ratio-4 compressor — coff 2, the overlapping window, ape on.
// ---------------------------------------------------------------------------

/// The overlapping `2·ratio` window with the position plane folded in. Two of
/// the six rows are chosen to stress the edges: one whose window runs off the
/// front of the sequence (`bpos = 3`, so four of the eight positions are
/// skipped) and one the boundary mark masked out, which must land zeros.
#[test]
fn the_ratio_four_pool_is_the_host_arithmetic() {
    let _serial = serialized();
    let Some(_device) = device_or_skip("the ratio-4 gather") else {
        return;
    };
    let fx = Fixture::new(
        0x9001_0001,
        4,
        vec![3, 7, 255, -1, 131, 63],
        vec![0, 0, 1, 0, 1, 1],
    );
    let got = fire(&fx, true);
    let want = fx.reference(true);

    // The masked row is the claim the band cannot make: it is zeros exactly.
    for d in 0..HEAD_DIM {
        assert_eq!(
            got[3 * HEAD_DIM + d],
            0.0,
            "the boundary mark masked row 3 out and the entry is not zero at lane {d}"
        );
    }
    let worst = worst_in_quanta(&got, &want);
    println!(
        "(a) pool_gather ratio 4 (coff 2, window 8, ape on): {} entries x {HEAD_DIM} lanes, \
         worst {worst:.3} bf16 quanta",
        fx.rows()
    );
    assert!(
        worst < 0.75,
        "the ratio-4 pool drifted {worst:.3} bf16 quanta — a window or a column, not rounding"
    );
}

// ---------------------------------------------------------------------------
// Gate (b): the ratio-128 compressor — coff 1, ape off.
// ---------------------------------------------------------------------------

/// The plain window at the artifact's other ratio, with NO position plane —
/// the CUDA `ape == nullptr` path, which on Metal is the unread seat plus
/// `has_ape = 0`. `bpos = 63` runs half the 128-wide window off the front.
#[test]
fn the_ratio_128_pool_is_the_host_arithmetic_without_a_position_plane() {
    let _serial = serialized();
    let Some(_device) = device_or_skip("the ratio-128 gather") else {
        return;
    };
    let fx = Fixture::new(
        0x9001_0002,
        128,
        vec![127, 255, 63, -1, 191],
        vec![0, 0, 1, 1, 1],
    );
    let got = fire(&fx, false);
    let want = fx.reference(false);
    let worst = worst_in_quanta(&got, &want);
    println!(
        "(b) pool_gather ratio 128 (coff 1, window 128, ape off): {} entries x {HEAD_DIM} lanes, \
         worst {worst:.3} bf16 quanta",
        fx.rows()
    );
    assert!(
        worst < 0.75,
        "the ratio-128 pool drifted {worst:.3} bf16 quanta"
    );
}

// ---------------------------------------------------------------------------
// Gates (c) and (d): the `has_ape` flag, in both directions.
// ---------------------------------------------------------------------------

/// **A ZERO POSITION PLANE IS THE ABSENT ONE, AND ANYTHING ELSE IS NOT.**
/// `has_ape` is the only thing separating the two calls in the first half, so
/// a shell that stated the flag and bound nothing — or bound the plane and
/// never stated the flag — fails one of these two.
#[test]
fn the_position_plane_is_read_exactly_when_the_flag_says_so() {
    let _serial = serialized();
    let Some(_device) = device_or_skip("the ape flag") else {
        return;
    };
    let mut fx = Fixture::new(0x9001_0003, 4, vec![7, 11, 255], vec![0, 0, 1]);

    // (c) an ape plane of zeros, read, against no plane at all.
    let zeroed = {
        let mut zeroed = fx.clone();
        zeroed.ape.iter_mut().for_each(|v| *v = 0.0);
        zeroed
    };
    let with_zeros = fire(&zeroed, true);
    let without = fire(&zeroed, false);
    assert_eq!(
        with_zeros, without,
        "a zeroed position plane changed the pool, so `has_ape` is selecting more than the fold"
    );
    println!("(c) has_ape over a zeroed plane: identical, {} entries", fx.rows());

    // (d) and a plane that is not zero moves the answer — on at least one
    // lane of every unmasked entry, not merely somewhere in the buffer.
    fx.ape.iter_mut().enumerate().for_each(|(k, v)| {
        *v = ((k % 7) as f32) - 3.0;
    });
    let folded = fire(&fx, true);
    let plain = fire(&fx, false);
    for c in 0..fx.rows() {
        let moved = (0..HEAD_DIM)
            .any(|d| folded[c * HEAD_DIM + d] != plain[c * HEAD_DIM + d]);
        assert!(
            moved,
            "entry {c} is unchanged by a position plane spanning six units of gate logit, \
             so the plane is bound and not read"
        );
    }
    println!("(d) has_ape over a live plane: every entry moved");
}

// ---------------------------------------------------------------------------
// Gate (e): the ROUND TRIP — the compressor's own writer feeding its reader.
// ---------------------------------------------------------------------------

/// One `attention.pool_state_write` followed by one `attention.pool_gather`,
/// in one command buffer, over zeroed slabs.
///
/// `tokens` is `(request, position)` per written row — the write descriptors
/// are derived from the same shuffled page map the gather walks, so the two
/// halves agree about a cell only if BOTH do the page-table arithmetic. A
/// writer that took `slot = row` would scatter into cells the gather never
/// looks at and every entry would come back zero.
fn round_trip(fx: &Fixture, tokens: &[(usize, usize)], has_ape: bool) -> Vec<f32> {
    let device = Context::bind().expect("the device binds");
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let width = fx.width;
    let n = tokens.len();
    // The projections, one row per written token, in the state's own element.
    let mut rng = Lcg(0x9001_00e5);
    let kv = rng.bf16_plane(n * width);
    let score = rng.bf16_plane(n * width);

    let w_page: Vec<u32> = tokens
        .iter()
        .map(|(req, pos)| PAGE_INDICES[req * PAGES_PER_REQ + pos / PAGE_SIZE])
        .collect();
    let w_off: Vec<u32> = tokens.iter().map(|(_, pos)| (pos % PAGE_SIZE) as u32).collect();

    let kv_buf = staged(&device, &encode_bf16(&kv));
    let score_buf = staged(&device, &encode_bf16(&score));
    let page_buf = staged(&device, &encode_u32(&w_page));
    let off_buf = staged(&device, &encode_u32(&w_off));
    // **ZEROED**, so every non-zero byte the gather reads came through the
    // writer and through nothing else.
    let state_kv_buf =
        Buffer::zeroed(&device, (CELLS * width * 2) as u64).expect("state_kv reserves");
    let state_score_buf =
        Buffer::zeroed(&device, (CELLS * width * 2) as u64).expect("state_score reserves");
    let ape_buf = staged(&device, &encode_f32(&fx.ape));
    let bpos = staged(&device, &encode_i32(&fx.bpos));
    let breq = staged(&device, &encode_i32(&fx.breq));
    let indices = staged(&device, &encode_u32(&PAGE_INDICES));
    let indptr = staged(&device, &encode_u32(&page_indptr()));
    let pages = Buffer::zeroed(&device, (CELLS * HEAD_DIM * 2) as u64).expect("the pool reserves");
    let out =
        Buffer::zeroed(&device, (fx.rows() * HEAD_DIM * 2) as u64).expect("the entries reserve");

    let cells = CELLS as u32;
    let width_u = width as u32;
    let pool = KvPool {
        keys: Tensor::new(
            bind_whole(&handles, &pages, "the pool keys"),
            cells,
            HEAD_DIM as u32,
            Dtype::Bf16,
        ),
        values: Tensor::new(
            bind_whole(&handles, &pages, "the pool values"),
            cells,
            HEAD_DIM as u32,
            Dtype::Bf16,
        ),
        page_indices: Tensor::new(
            bind_whole(&handles, &indices, "the page map"),
            PAGE_INDICES.len() as u32,
            1,
            Dtype::U32,
        ),
        page_indptr: Tensor::new(
            bind_whole(&handles, &indptr, "the page spans"),
            (REQUESTS + 1) as u32,
            1,
            Dtype::U32,
        ),
        page_size: PAGE_SIZE as i32,
        seq_stride: HEAD_DIM as u64,
        head_stride: HEAD_DIM as u64,
    };
    let state_kv = Tensor::new(
        bind_whole(&handles, &state_kv_buf, "state_kv"),
        cells,
        width_u,
        Dtype::Bf16,
    );
    let state_score = Tensor::new(
        bind_whole(&handles, &state_score_buf, "state_score"),
        cells,
        width_u,
        Dtype::Bf16,
    );

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::pool::state_write(
            &sink,
            Tensor::new(bind_whole(&handles, &kv_buf, "wkv.x"), n as u32, width_u, Dtype::Bf16),
            Tensor::new(
                bind_whole(&handles, &score_buf, "wgate.x"),
                n as u32,
                width_u,
                Dtype::Bf16,
            ),
            &pool,
            Tensor::new(bind_whole(&handles, &page_buf, "write_page"), n as u32, 1, Dtype::U32),
            Tensor::new(bind_whole(&handles, &off_buf, "write_offset"), n as u32, 1, Dtype::U32),
            HEAD_DIM as u32,
            fx.ratio as u32,
            state_kv,
            state_score,
        )
        .expect("the state write encodes");
        kernels_metal::attn::pool::gather(
            &sink,
            Tensor::new(
                bind_whole(&handles, &bpos, "boundary_pos"),
                fx.rows() as u32,
                1,
                Dtype::I32,
            ),
            Tensor::new(
                bind_whole(&handles, &breq, "boundary_req"),
                fx.rows() as u32,
                1,
                Dtype::I32,
            ),
            &pool,
            HEAD_DIM as u32,
            fx.ratio as u32,
            state_kv,
            state_score,
            has_ape.then(|| {
                Tensor::new(
                    bind_whole(&handles, &ape_buf, "the position plane"),
                    fx.ratio as u32,
                    width_u,
                    Dtype::F32,
                )
            }),
            Tensor::new(
                bind_whole(&handles, &out, "the pooled entries"),
                fx.rows() as u32,
                HEAD_DIM as u32,
                Dtype::Bf16,
            ),
        )
        .expect("the gather encodes");
    }
    frame.commit().expect("the round trip completes");
    decode_bf16(&read_back(&out, fx.rows() * HEAD_DIM * 2))
}

/// **THE STATE SLABS FINALLY HAVE A WRITER, AND THIS IS THE ONE CLAIM THAT
/// NEEDED ONE.** Every gate above hand-built the state plane and asked
/// whether the gather read it correctly; none of them could ask whether
/// anything in the stack could PUT numbers there. The compressor's two
/// projections now do (`attention.pool_state_write`), and the two halves
/// address a cell through the same shuffled page map from opposite ends —
/// the writer off `write_page`/`write_offset`, the reader off
/// `(request, position)` and the page table.
///
/// The reference stages the identical scatter on the host and then runs the
/// same gated pool the gates above run, so a disagreement is the ADDRESSING
/// and not the arithmetic.
#[test]
fn what_the_compressor_writes_is_what_the_pool_gathers() {
    let _serial = serialized();
    let Some(_device) = device_or_skip("the compressor round trip") else {
        return;
    };

    // One request's first forty positions — across two pages, so the write
    // descriptors are not all the same page and the map is really walked.
    let tokens: Vec<(usize, usize)> = (0..40).map(|pos| (0usize, pos)).collect();
    // Boundaries whose windows lie entirely inside what was written: the
    // ratio-4 window is eight positions back from the mark.
    let mut fx = Fixture::new(0x9001_0005, 4, vec![7, 15, 39, -1], vec![0, 0, 0, 0]);
    // The state planes this fixture generated are not the ones under test —
    // the WRITER supplies those — so the reference's copies are replaced by
    // the scatter below and the fixture keeps only its boundary tables and
    // its position plane.
    let width = fx.width;
    let mut rng = Lcg(0x9001_00e5);
    let kv = rng.bf16_plane(tokens.len() * width);
    let score = rng.bf16_plane(tokens.len() * width);
    fx.state_kv = vec![0.0; CELLS * width];
    fx.state_score = vec![0.0; CELLS * width];
    for (row, (req, pos)) in tokens.iter().enumerate() {
        let slot = paged_slot(*req, *pos);
        for d in 0..width {
            fx.state_kv[slot * width + d] = kv[row * width + d];
            fx.state_score[slot * width + d] = score[row * width + d];
        }
    }

    let got = round_trip(&fx, &tokens, true);
    let want = fx.reference(true);

    // The written state is not all zeros, so a gather that read nothing would
    // be caught by the band rather than agreeing with an empty reference.
    let live = want.iter().any(|v| *v != 0.0);
    assert!(live, "the reference pooled nothing, so this gate proves nothing");

    for d in 0..HEAD_DIM {
        assert_eq!(
            got[3 * HEAD_DIM + d],
            0.0,
            "the boundary mark masked row 3 out and the entry is not zero at lane {d}"
        );
    }
    let worst = worst_in_quanta(&got, &want);
    println!(
        "(e) pool_state_write -> pool_gather round trip: {} written rows, {} entries x \
         {HEAD_DIM} lanes, worst {worst:.3} bf16 quanta",
        tokens.len(),
        fx.rows()
    );
    assert!(
        worst < 0.75,
        "the round trip drifted {worst:.3} bf16 quanta — a cell, not rounding"
    );
}

// ---------------------------------------------------------------------------
// Gate (f): the boundary marks, and the two DIFFERENT positions of one entry.
// ---------------------------------------------------------------------------

/// **AN ENTRY HAS A CELL AND AN ANGLE AND THEY ARE NOT THE SAME NUMBER.**
///
/// Every gate above this one takes the boundary tables as GIVEN — the fixture
/// types `bpos` and the kernels are asked what they pool at those marks. So
/// nothing measured what produces the marks, and in particular nothing
/// measured the second column the compressor needs: a pooled entry is CACHED
/// at the cell its window closes on (`(c+1)·ratio - 1` — `3, 7, 11, …` at
/// ratio 4, which is what `pool_lse` reads back) and ROPED at the compressed
/// row's own position (`c·ratio` — `0, 4, 8, …`). The reference is explicit:
/// `compressor_prefill` ropes the pooled plane at
/// `rows = mx.arange(0, cutoff, ratio)` (`v4mlx/compressor.py`), the block
/// STARTS.
///
/// The model text roped at the raw token positions, which agree with the
/// closing cell on every boundary row — so every compressed key it cached,
/// the attention branch's and the indexer's alike, carried an angle exactly
/// `ratio - 1` positions too far. This gate is the column that says so: the
/// old kernels published no third output at all, and against this reference
/// what they left the text reading scores `3, 7, 11, 15, 19` where the
/// reference wants `0, 4, 8, 12, 16` — every entry wrong, none of them
/// detectably so from the shapes.
#[test]
fn the_boundary_marks_publish_the_cell_and_the_compressed_row() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the boundary marks") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    const RATIO: i32 = 4;
    // Two lanes' worth of a ragged prefill: 12 positions then 9, and a
    // trailing graph-padding row the mask must drop even though it closes a
    // boundary by arithmetic.
    let positions: Vec<i32> = (0..12).chain(0..9).chain(std::iter::once(3)).collect();
    let qo_indptr: Vec<u32> = vec![0, 12, 21, 22];
    let mut valid = vec![1u8; positions.len()];
    *valid.last_mut().expect("a last row") = 0;
    let rows = positions.len() as u32;

    let pos_buf = staged(&device, &encode_i32(&positions));
    let indptr_buf = staged(&device, &encode_u32(&qo_indptr));
    let valid_buf = staged(&device, &valid);
    let out_pos = Buffer::zeroed(&device, (rows as usize * 4) as u64).expect("cells reserve");
    let out_req = Buffer::zeroed(&device, (rows as usize * 4) as u64).expect("lanes reserve");
    let out_rope = Buffer::zeroed(&device, (rows as usize * 4) as u64).expect("angles reserve");

    let i32t = |b: &Buffer, what: &str| Tensor::new(bind_whole(&handles, b, what), rows, 1, Dtype::I32);

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::pool::boundary_prefill(
            &sink,
            kernels_metal::RaggedTensor {
                data: i32t(&pos_buf, "positions"),
                indptr: Tensor::new(
                    bind_whole(&handles, &indptr_buf, "qo_indptr"),
                    qo_indptr.len() as u32,
                    1,
                    Dtype::U32,
                ),
            },
            Tensor::new(bind_whole(&handles, &valid_buf, "row_valid"), rows, 1, Dtype::U8),
            RATIO as u32,
            i32t(&out_pos, "boundary_pos"),
            i32t(&out_req, "boundary_req"),
            i32t(&out_rope, "boundary_rope"),
        )
        .expect("the boundary mark encodes");
    }
    frame.commit().expect("the boundary mark completes");

    let got_pos = decode_i32(&read_back(&out_pos, rows as usize * 4));
    let got_req = decode_i32(&read_back(&out_req, rows as usize * 4));
    let got_rope = decode_i32(&read_back(&out_rope, rows as usize * 4));

    // The host reference, said the reference's way: within each lane the
    // boundaries are the closing tokens, and their rope positions are that
    // lane's `arange(0, cutoff, ratio)`.
    let mut want_pos = vec![-1i32; rows as usize];
    let mut want_req = vec![0i32; rows as usize];
    let mut want_rope = vec![0i32; rows as usize];
    let mut arange: Vec<i32> = Vec::new();
    for lane in 0..qo_indptr.len() - 1 {
        let (lo, hi) = (qo_indptr[lane] as usize, qo_indptr[lane + 1] as usize);
        let cutoff = (hi - lo) as i32 - ((hi - lo) as i32 % RATIO);
        let mut starts: Vec<i32> = (0..cutoff).step_by(RATIO as usize).collect();
        for t in lo..hi {
            want_req[t] = lane as i32;
            if valid[t] == 0 {
                continue;
            }
            let p = positions[t];
            if (p + 1) % RATIO == 0 {
                want_pos[t] = p;
                want_rope[t] = (p / RATIO) * RATIO;
            }
        }
        // Only the valid lanes state an arange; the padding lane closes none.
        if valid[lo] != 0 {
            arange.append(&mut starts);
        }
    }

    assert_eq!(got_req, want_req, "the lane column");
    assert_eq!(got_pos, want_pos, "the cell column");
    assert_eq!(got_rope, want_rope, "the compressed-row column");

    // And the claim in the reference's own words: the rope column, read at the
    // rows that closed a boundary, IS `arange(0, cutoff, ratio)` per lane.
    let fired: Vec<i32> = (0..rows as usize)
        .filter(|t| got_pos[*t] >= 0)
        .map(|t| got_rope[t])
        .collect();
    let cells: Vec<i32> = (0..rows as usize)
        .filter(|t| got_pos[*t] >= 0)
        .map(|t| got_pos[t])
        .collect();
    assert_eq!(fired, arange, "the rope column is arange(0, cutoff, ratio)");
    assert_eq!(cells, [3, 7, 11, 3, 7], "the cell column is the closing token");
    assert_eq!(fired, [0, 4, 8, 0, 4], "and the rope column is the block start");
    for (cell, angle) in cells.iter().zip(&fired) {
        assert_eq!(cell - angle, RATIO - 1, "the two positions are ratio-1 apart");
    }

    // The decode twin, over the same claim: one row per lane, its own request.
    let dec_pos: Vec<i32> = vec![3, 6, 127];
    let dec_rows = dec_pos.len() as u32;
    let dec_buf = staged(&device, &encode_i32(&dec_pos));
    let dec_valid = staged(&device, &vec![1u8; dec_pos.len()]);
    let d_pos = Buffer::zeroed(&device, (dec_rows as usize * 4) as u64).expect("cells reserve");
    let d_req = Buffer::zeroed(&device, (dec_rows as usize * 4) as u64).expect("lanes reserve");
    let d_rope = Buffer::zeroed(&device, (dec_rows as usize * 4) as u64).expect("angles reserve");
    let d32 = |b: &Buffer, what: &str| {
        Tensor::new(bind_whole(&handles, b, what), dec_rows, 1, Dtype::I32)
    };
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::pool::boundary_decode(
            &sink,
            d32(&dec_buf, "positions"),
            Tensor::new(bind_whole(&handles, &dec_valid, "row_valid"), dec_rows, 1, Dtype::U8),
            RATIO as u32,
            d32(&d_pos, "boundary_pos"),
            d32(&d_req, "boundary_req"),
            d32(&d_rope, "boundary_rope"),
        )
        .expect("the decode mark encodes");
    }
    frame.commit().expect("the decode mark completes");
    assert_eq!(decode_i32(&read_back(&d_pos, dec_rows as usize * 4)), [3, -1, 127]);
    assert_eq!(decode_i32(&read_back(&d_rope, dec_rows as usize * 4)), [0, 0, 124]);

    println!(
        "(f) boundary marks: {rows} prefill rows over 3 lanes + {dec_rows} decode rows; \
         cells {cells:?}, compressed rows {fired:?} — ratio-1 apart, and the rope column is \
         arange(0, cutoff, {RATIO})"
    );
}

// ---------------------------------------------------------------------------
// Host staging — `hc_on_device`'s helpers, for its reasons.
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

    /// `n` values in `[-0.25, 0.25)`, **already through bf16**.
    fn bf16_plane(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32_of(bf16_bits(self.next_f32()))).collect()
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

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The bf16 quantum at `v`: eight significant bits below the binade.
fn quantum(v: f32) -> f32 {
    if v == 0.0 {
        return f32::MIN_POSITIVE;
    }
    v.abs().log2().floor().exp2() / 128.0
}

fn staged(device: &Context, bytes: &[u8]) -> Buffer {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("the reservation lands");
    buffer.write(0, bytes).expect("the bytes land");
    buffer
}

fn bind_whole(handles: &Handles, buffer: &Buffer, what: &str) -> u32 {
    handles
        .bind(buffer, 0, buffer.bytes())
        .unwrap_or_else(|fault| panic!("{what} binds: {fault}"))
}

fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut got = vec![0u8; bytes];
    buffer.read(0, &mut got).expect("the answer reads back");
    got
}
