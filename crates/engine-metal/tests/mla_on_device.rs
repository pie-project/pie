//! The absorbed multi-head latent attention family, on a real Apple GPU: the
//! paged latent appender, the naive flash reader in both its dense and its
//! index-selected forms, and — the reason this file exists — the pair of
//! absorbs, measured against the UNABSORBED attention they claim to equal.
//!
//! **WHAT THIS IS FOR, AND WHAT IT SETTLED.** `attention.mla_absorb_out` was
//! the last fatal MLA refusal on this plane, and it was deferred over one
//! number: where `kv_b`'s value-up planes begin. `kernels-cuda`'s
//! `attn/mla.rs` starts the GEMM's A operand at
//! `kv_b.ptr.wrapping_add(2 * nope * rank)`, and read as an ELEMENT count
//! that is not the standard DeepSeek packing — it lands a whole `nope` block
//! past where the value rows should be. The previous lane refused to guess
//! and left the entry unsupported, which was the right call and the wrong
//! stopping point: the reconciliation is that `Tensor::ptr` is a raw device
//! ADDRESS and `wrapping_add` on it is BYTE arithmetic. `kernels-cuda`'s own
//! neighbouring guard is written in the same units (`plane_bytes =
//! rows * width * 2`). The `2` is `sizeof(bf16)`. The value block begins
//! `nope * rank` ELEMENTS in, which IS the standard packing:
//!
//! ```text
//! kv_b : row-major [heads * (nope + v_dim), rank]
//!   head h owns rows [h*(nope+v_dim) .. +(nope+v_dim))
//!     rows [0 .. nope)             = W_UK[h], the key-up block   (mla_absorb_q)
//!     rows [nope .. nope + v_dim)  = W_UV[h], the value-up block (mla_absorb_out)
//!   heads OUTER, the two blocks contiguous within a head, each row `rank` wide
//! ```
//!
//! Reading is not measuring, so [`the_absorbed_pair_is_the_unabsorbed_attention`]
//! does not assert the packing — it SWEEPS the three candidate bases
//! (`0`, `nope*rank`, `2*nope*rank`) against a CPU reference of the
//! unabsorbed computation and reports the max abs difference of each. Exactly
//! one lands inside the bf16 accumulation band and the other two are off by
//! two orders of magnitude, which is the shape the deferral predicted: a
//! wrong base is garbage, not epsilon.
//!
//! # The four gates
//!
//! ```text
//! (a) mla_kv_append          — the latent pool is FILLED BY THE DEVICE, at the
//!                              slots the write tables name, and the pages no
//!                              table names stay zero
//! (b) mla_naive_paged        — the dense reader vs a CPU online-softmax over
//!                              [0, position], across page boundaries
//! (c) mla_naive_paged_selected — the same engine handed an index row with a
//!                              -1 padded tail AND an out-of-bound entry, vs
//!                              the CPU reference restricted to the keys that
//!                              survive the sweep
//! (d) absorb_q + flash + absorb_out — the whole absorbed chain vs a CPU
//!                              reference of the UNABSORBED attention, plus the
//!                              V-block candidate sweep
//! ```
//!
//! and one sanity instantiation at dsv4/glm5's real geometry (H=64, CKV=512,
//! KPE=64), because a shape that only ever runs at H=4 has not been fired.
//!
//! **THE FIRES GO THROUGH THE SHIPPED RUST ENCODERS**, as `device_floor` and
//! `tower_kernels_on_device` do — half of what these gates can catch is a
//! host-side launch shape, and a test that bound the buffers itself would
//! only check the shader against the test's own idea of the signature.
//!
//! # Gating
//!
//! As `device_floor`: `cfg`'d to Apple at compile time, and SKIPS at run time
//! when `device::present()` says no, saying so.
//!
//! ```text
//! cargo test -p engine-metal --release --test mla_on_device -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{KvPool, Tensor};
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, for `device_floor`'s reason: `cargo test` runs a
/// file's tests on several threads, each of these binds a device and reserves
/// buffers, and two of them compiling shaders at once is a way to meet the
/// Metal compiler's own concurrency and learn nothing.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The device, or a printed skip and `None`.
fn device_or_skip(what: &str) -> Option<Context> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    Some(Context::bind().expect("the device binds"))
}

// ---------------------------------------------------------------------------
// The fixture: a small latent pool with page boundaries and a non-identity
// page table, and every tensor the family reads.
// ---------------------------------------------------------------------------

const HEADS: usize = 4;
/// The latent rank (`ckv`), two 32-lane strips — enough that the flash body's
/// per-lane loop runs more than once.
const RANK: usize = 64;
/// The rope width (`kpe`), also two strips.
const KPE: usize = 64;
/// `q_nope`'s per-head width, the axis `absorb_q` contracts.
const NOPE: usize = 32;
/// The value head dim, the axis `absorb_out` produces. Deliberately NOT equal
/// to `NOPE`, so a candidate base off by one block is off by a distinguishable
/// amount rather than landing on a block boundary.
const VDIM: usize = 64;

const PAGE: usize = 4;
const POOL_PAGES: usize = 8;

/// **A NON-IDENTITY PAGE TABLE**, so a kernel that ignored `kv_page_indices`
/// and addressed the pool linearly would read the wrong slots rather than the
/// right ones by luck. Pages 0, 3 and 4 are named by nobody and must stay zero.
const PAGE_INDICES: [u32; 5] = [5, 2, 7, 1, 6];
const PAGE_INDPTR: [u32; 3] = [0, 3, 5];
/// Cached keys per request: 10 spans three pages with a partial last one, 7
/// spans two.
const KV_LEN: [usize; 2] = [10, 7];

/// The query rows: `(request, causal position)`. Row 0 is a decode at the last
/// cached slot, row 1 a mid-sequence row whose sweep stops INSIDE the second
/// page, row 2 a decode on the other request, row 3 a row whose sweep stops
/// exactly at a page boundary.
const ROWS: [(usize, i32); 4] = [(0, 9), (0, 4), (1, 6), (1, 3)];

const TOPK: usize = 5;
/// Ascending key ids with a -1 padded tail — and row 1 carries `6`, which is
/// past its own causal bound of 5, so the sweep's `continue` is exercised on
/// something other than a pad.
const SELECTION: [[i32; TOPK]; ROWS.len()] = [
    [0, 3, 7, 9, -1],
    [1, 4, 6, -1, -1],
    [0, 2, 4, 5, 6],
    [3, -1, -1, -1, -1],
];

/// Cached tokens, flattened across requests in request order.
const CACHED: usize = KV_LEN[0] + KV_LEN[1];

fn sm_scale() -> f32 {
    1.0 / ((RANK + KPE) as f32).sqrt()
}

/// Every host-side tensor, already through bf16 — so the device and the
/// reference are reading the SAME numbers and the only difference either can
/// show is arithmetic.
struct Fixture {
    /// `[CACHED, RANK]` — the latent keys, which are also the values.
    ckv: Vec<f32>,
    /// `[CACHED, KPE]` — the rope tails.
    kpe: Vec<f32>,
    /// `[rows, HEADS, RANK]` — an already-absorbed query, for gates (b)/(c).
    q_lat: Vec<f32>,
    /// `[rows, HEADS, KPE]`.
    q_pe: Vec<f32>,
    /// `[rows, HEADS, NOPE]` — the unabsorbed query, for gate (d).
    q_nope: Vec<f32>,
    /// `[HEADS*(NOPE+VDIM) + NOPE, RANK]`.
    ///
    /// **THE TAIL IS NOT WEIGHT.** The device only ever reads the first
    /// `HEADS*(NOPE+VDIM)` rows. The extra `NOPE` rows exist so the candidate
    /// base `2*nope*rank` can be EVALUATED on the host for the last head
    /// without running off the end — a candidate that cannot be computed
    /// cannot be ruled out, and ruling it out is the point.
    kv_b: Vec<f32>,
}

/// The flat index of request `r`'s key `j` among the cached tokens.
fn cached_index(r: usize, j: usize) -> usize {
    KV_LEN[..r].iter().sum::<usize>() + j
}

/// The pool slot request `r`'s key `j` lands in, by the page table.
fn slot_of(r: usize, j: usize) -> usize {
    let page = PAGE_INDICES[PAGE_INDPTR[r] as usize + j / PAGE] as usize;
    page * PAGE + j % PAGE
}

impl Fixture {
    fn new(seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let rows = ROWS.len();
        Self {
            ckv: rng.plane(CACHED * RANK),
            kpe: rng.plane(CACHED * KPE),
            q_lat: rng.plane(rows * HEADS * RANK),
            q_pe: rng.plane(rows * HEADS * KPE),
            q_nope: rng.plane(rows * HEADS * NOPE),
            kv_b: rng.plane((HEADS * (NOPE + VDIM) + NOPE) * RANK),
        }
    }

    fn ckv_row(&self, r: usize, j: usize) -> &[f32] {
        let c = cached_index(r, j);
        &self.ckv[c * RANK..(c + 1) * RANK]
    }

    fn kpe_row(&self, r: usize, j: usize) -> &[f32] {
        let c = cached_index(r, j);
        &self.kpe[c * KPE..(c + 1) * KPE]
    }

    fn q_lat_row(&self, row: usize, h: usize) -> &[f32] {
        let base = (row * HEADS + h) * RANK;
        &self.q_lat[base..base + RANK]
    }

    fn q_pe_row(&self, row: usize, h: usize) -> &[f32] {
        let base = (row * HEADS + h) * KPE;
        &self.q_pe[base..base + KPE]
    }

    fn q_nope_row(&self, row: usize, h: usize) -> &[f32] {
        let base = (row * HEADS + h) * NOPE;
        &self.q_nope[base..base + NOPE]
    }

    /// Row `k` of head `h`'s key-up block: `W_UK[h][k]`, `RANK` wide.
    fn w_uk(&self, h: usize, k: usize) -> &[f32] {
        let base = (h * (NOPE + VDIM) + k) * RANK;
        &self.kv_b[base..base + RANK]
    }

    /// Row `k` of head `h`'s value-up block AT A CANDIDATE BASE: the block
    /// starts `blocks * NOPE * RANK` elements past head `h`'s own base.
    /// `blocks == 1` is the shader's reading.
    fn w_uv_at(&self, h: usize, k: usize, blocks: usize) -> &[f32] {
        let base = h * (NOPE + VDIM) * RANK + blocks * NOPE * RANK + k * RANK;
        &self.kv_b[base..base + RANK]
    }

    /// The absorbed query one head of one row reads, in f32 — the arithmetic
    /// `mla_absorb_q` performs, which this file has no separate gate for
    /// because gate (d) fires the real one.
    fn absorbed_q(&self, row: usize, h: usize) -> Vec<f32> {
        let q = self.q_nope_row(row, h);
        (0..RANK)
            .map(|i| (0..NOPE).map(|n| q[n] * self.w_uk(h, n)[i]).sum())
            .collect()
    }
}

/// The key ids a row's dense sweep visits: `[0, position]`.
fn dense_keys(row: usize) -> Vec<usize> {
    (0..=ROWS[row].1 as usize).collect()
}

/// The key ids a row's SELECTED sweep visits — `mla::selected_sweep`'s rule,
/// stated here from the other side: every entry outside `[0, j_end)` is
/// dropped, the -1 pads and the over-bound entry alike.
fn selected_keys(row: usize) -> Vec<usize> {
    let j_end = ROWS[row].1 + 1;
    SELECTION[row]
        .iter()
        .filter(|&&j| j >= 0 && j < j_end)
        .map(|&j| j as usize)
        .collect()
}

// ---------------------------------------------------------------------------
// The CPU references.
// ---------------------------------------------------------------------------

/// A batch softmax reading, in f32: the answer the shader's streaming
/// online-softmax must equal.
fn reading(scores: &[f32], values: &[Vec<f32>], width: usize) -> Vec<f32> {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
    let lsum: f32 = exps.iter().sum();
    let inv = if lsum > 0.0 { 1.0 / lsum } else { 0.0 };
    (0..width)
        .map(|i| exps.iter().zip(values).map(|(p, v)| p * v[i]).sum::<f32>() * inv)
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The latent-space reading `mla_naive_paged` computes for one `(row, head)`
/// over a given key set: scores from the absorbed query, values the latents
/// themselves.
fn cpu_latent_reading(fx: &Fixture, row: usize, h: usize, q_lat: &[f32], keys: &[usize]) -> Vec<f32> {
    let (r, _) = ROWS[row];
    let scale = sm_scale();
    let qp = fx.q_pe_row(row, h);
    let scores: Vec<f32> = keys
        .iter()
        .map(|&j| (dot(q_lat, fx.ckv_row(r, j)) + dot(qp, fx.kpe_row(r, j))) * scale)
        .collect();
    let values: Vec<Vec<f32>> = keys.iter().map(|&j| fx.ckv_row(r, j).to_vec()).collect();
    reading(&scores, &values, RANK)
}

/// **THE UNABSORBED ATTENTION**, which is the claim gate (d) is about: the
/// key is `[W_UK[h] · ckv_j ; kpe_j]` and the value is `W_UV[h] · ckv_j`, both
/// materialized per key, with NO latent-space shortcut anywhere. `blocks`
/// selects which candidate base the value planes are read at.
fn cpu_unabsorbed(fx: &Fixture, row: usize, h: usize, keys: &[usize], blocks: usize) -> Vec<f32> {
    let (r, _) = ROWS[row];
    let scale = sm_scale();
    let qn = fx.q_nope_row(row, h);
    let qp = fx.q_pe_row(row, h);
    let mut scores = Vec::with_capacity(keys.len());
    let mut values = Vec::with_capacity(keys.len());
    for &j in keys {
        let ckv = fx.ckv_row(r, j);
        let k_nope: Vec<f32> = (0..NOPE).map(|n| dot(fx.w_uk(h, n), ckv)).collect();
        scores.push((dot(qn, &k_nope) + dot(qp, fx.kpe_row(r, j))) * scale);
        values.push((0..VDIM).map(|v| dot(fx.w_uv_at(h, v, blocks), ckv)).collect());
    }
    reading(&scores, &values, VDIM)
}

// ---------------------------------------------------------------------------
// Device staging.
// ---------------------------------------------------------------------------

/// The pool's four reservations, held so their handles stay resolvable.
struct Pool {
    keys: Buffer,
    kpe: Buffer,
    indices: Buffer,
    indptr: Buffer,
}

impl Pool {
    /// The pages ZEROED — gate (a)'s claim about untouched slots rests on it.
    fn reserve(device: &Context) -> Self {
        let slots = (POOL_PAGES * PAGE) as u64;
        Self {
            keys: Buffer::zeroed(device, slots * RANK as u64 * 2).expect("the ckv pages reserve"),
            kpe: Buffer::zeroed(device, slots * KPE as u64 * 2).expect("the kpe pages reserve"),
            indices: staged(device, as_bytes_u32(&PAGE_INDICES)),
            indptr: staged(device, as_bytes_u32(&PAGE_INDPTR)),
        }
    }

    fn view(&self, handles: &Handles) -> KvPool {
        let slots = (POOL_PAGES * PAGE) as u32;
        KvPool {
            keys: Tensor::new(bind_whole(handles, &self.keys, "the ckv pages"), slots, RANK as u32, Dtype::Bf16),
            values: Tensor::new(bind_whole(handles, &self.kpe, "the kpe pages"), slots, KPE as u32, Dtype::Bf16),
            page_indices: Tensor::new(
                bind_whole(handles, &self.indices, "the page table"),
                PAGE_INDICES.len() as u32,
                1,
                Dtype::U32,
            ),
            page_indptr: Tensor::new(
                bind_whole(handles, &self.indptr, "the page spans"),
                PAGE_INDPTR.len() as u32,
                1,
                Dtype::U32,
            ),
            page_size: PAGE as i32,
            seq_stride: RANK as u64,
            head_stride: RANK as u64,
        }
    }
}

/// The write tables gate (a) fires the appender over: one destination page and
/// one in-page slot per cached token, in the flattened request order.
fn write_tables() -> (Vec<u32>, Vec<u32>) {
    let mut pages = Vec::with_capacity(CACHED);
    let mut offsets = Vec::with_capacity(CACHED);
    for (r, &len) in KV_LEN.iter().enumerate() {
        for j in 0..len {
            pages.push(PAGE_INDICES[PAGE_INDPTR[r] as usize + j / PAGE]);
            offsets.push((j % PAGE) as u32);
        }
    }
    (pages, offsets)
}

/// Fills the pool THROUGH THE DEVICE — `mla_kv_append` at the write tables —
/// so every later gate reads pages the appender wrote.
fn fill_pool(device: &Context, pipelines: &Pipelines, handles: &Handles, pool: &Pool, fx: &Fixture) {
    let (pages, offsets) = write_tables();
    let kv_c = staged(device, &encode(&fx.ckv));
    let k_pe = staged(device, &encode(&fx.kpe));
    let wp = staged(device, as_bytes_u32(&pages));
    let wo = staged(device, as_bytes_u32(&offsets));

    let kv_c_h = bind_whole(handles, &kv_c, "the latent rows");
    let k_pe_h = bind_whole(handles, &k_pe, "the rope rows");
    let wp_h = bind_whole(handles, &wp, "the write pages");
    let wo_h = bind_whole(handles, &wo, "the write offsets");
    let view = pool.view(handles);

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        kernels_metal::attn::mla::kv_append(
            &sink,
            Tensor::new(kv_c_h, CACHED as u32, RANK as u32, Dtype::Bf16),
            Tensor::new(k_pe_h, CACHED as u32, KPE as u32, Dtype::Bf16),
            &view,
            Tensor::new(wp_h, CACHED as u32, 1, Dtype::U32),
            Tensor::new(wo_h, CACHED as u32, 1, Dtype::U32),
        )
        .expect("the latent append encodes");
    }
    frame.commit().expect("the append completes");
}

// ---------------------------------------------------------------------------
// (a) The pool is filled by the device, at the slots the tables name.
// ---------------------------------------------------------------------------

/// (a): `mla_kv_append` puts each latent row and its rope tail at
/// `write_page * page_size + write_offset`, and leaves every other slot alone.
///
/// The permutation is the claim — three of the eight pages are named by no
/// table, and a kernel that addressed the pool linearly would fill pages 0..4
/// and leave 5..7 empty, which is the exact opposite of what this asserts.
#[test]
fn the_latent_append_lands_every_row_at_the_slot_its_tables_name() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the latent append") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x51ee_d0a1);
    let pool = Pool::reserve(&device);
    fill_pool(&device, &pipelines, &handles, &pool, &fx);

    let ckv = decode(&read_back(&pool.keys, (POOL_PAGES * PAGE) * RANK * 2));
    let kpe = decode(&read_back(&pool.kpe, (POOL_PAGES * PAGE) * KPE * 2));

    let mut written = [false; POOL_PAGES * PAGE];
    for (r, &len) in KV_LEN.iter().enumerate() {
        for j in 0..len {
            let slot = slot_of(r, j);
            written[slot] = true;
            assert_eq!(
                &ckv[slot * RANK..(slot + 1) * RANK],
                fx.ckv_row(r, j),
                "request {r} key {j} did not land in slot {slot}"
            );
            assert_eq!(
                &kpe[slot * KPE..(slot + 1) * KPE],
                fx.kpe_row(r, j),
                "request {r} key {j}'s rope tail did not land in slot {slot}"
            );
        }
    }
    let untouched = written.iter().filter(|w| !**w).count();
    for (slot, _) in written.iter().enumerate().filter(|(_, w)| !**w) {
        assert!(
            ckv[slot * RANK..(slot + 1) * RANK].iter().all(|v| *v == 0.0),
            "slot {slot} is named by no write table and was written anyway"
        );
    }
    println!(
        "(a) mla_kv_append: {CACHED} rows landed byte-exact at their named slots; \
         {untouched} of {} slots untouched",
        POOL_PAGES * PAGE
    );
}

// ---------------------------------------------------------------------------
// (b) The dense reader.
// ---------------------------------------------------------------------------

/// (b): `mla_naive_paged` over `[0, position]`, against a CPU batch softmax
/// over the same keys.
///
/// The streaming online-softmax the shader runs and the batch softmax here are
/// the same function of the same numbers; what separates them is f32
/// accumulation order (the shader folds a 32-lane `simd_sum` tree) and
/// `fast::exp` against `f32::exp`. The band is stated in bf16 quanta of the
/// answer, which is the unit the store rounds to.
#[test]
fn the_dense_latent_reader_is_the_softmax_over_its_causal_prefix() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the dense latent reader") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x51ee_d0a1);
    let pool = Pool::reserve(&device);
    fill_pool(&device, &pipelines, &handles, &pool, &fx);

    let got = fire_flash(&device, &pipelines, &handles, &pool, &fx, &fx.q_lat, None);

    let mut worst = 0.0f32;
    let mut worst_quanta = 0.0f32;
    for (row, _) in ROWS.iter().enumerate() {
        for h in 0..HEADS {
            let want = cpu_latent_reading(&fx, row, h, fx.q_lat_row(row, h), &dense_keys(row));
            let base = (row * HEADS + h) * RANK;
            for (i, w) in want.iter().enumerate() {
                let g = got[base + i];
                worst = worst.max((g - w).abs());
                worst_quanta = worst_quanta.max((g - w).abs() / quantum(w.abs().max(0.05)));
            }
        }
    }
    println!("(b) mla_naive_paged: max abs diff {worst:.3e} ({worst_quanta:.2} bf16 quanta)");
    assert!(
        worst_quanta <= 2.0,
        "the dense reading drifted {worst_quanta:.2} quanta from the softmax it claims to be"
    );
}

// ---------------------------------------------------------------------------
// (c) The index-selected reader.
// ---------------------------------------------------------------------------

/// (c): the same engine handed an index row — a -1 padded tail on three rows
/// and, on row 1, a key id PAST that row's causal bound. The reference is the
/// batch softmax over exactly the keys the sweep's `continue` rule leaves.
///
/// **AND THE SELECTED ANSWER IS NOT THE DENSE ANSWER**, which is asserted too:
/// a kernel that ignored `selection` and swept `[0, j_end)` would pass every
/// tolerance above against the DENSE reference, so the gate also states that
/// the two readings differ by far more than the band.
#[test]
fn the_selected_latent_reader_attends_the_keys_its_index_row_names_and_no_others() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the selected latent reader") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x51ee_d0a1);
    let pool = Pool::reserve(&device);
    fill_pool(&device, &pipelines, &handles, &pool, &fx);

    let flat: Vec<i32> = SELECTION.iter().flatten().copied().collect();
    let got = fire_flash(&device, &pipelines, &handles, &pool, &fx, &fx.q_lat, Some(&flat));

    let mut worst = 0.0f32;
    let mut worst_quanta = 0.0f32;
    let mut apart = f32::INFINITY;
    for (row, _) in ROWS.iter().enumerate() {
        let keys = selected_keys(row);
        assert!(
            keys.len() < dense_keys(row).len(),
            "row {row}'s selection must be a PROPER subset, or the gate proves nothing"
        );
        for h in 0..HEADS {
            let q = fx.q_lat_row(row, h);
            let want = cpu_latent_reading(&fx, row, h, q, &keys);
            let dense = cpu_latent_reading(&fx, row, h, q, &dense_keys(row));
            let base = (row * HEADS + h) * RANK;
            let mut this_apart = 0.0f32;
            for (i, w) in want.iter().enumerate() {
                let g = got[base + i];
                worst = worst.max((g - w).abs());
                worst_quanta = worst_quanta.max((g - w).abs() / quantum(w.abs().max(0.05)));
                this_apart = this_apart.max((g - dense[i]).abs());
            }
            apart = apart.min(this_apart);
        }
    }
    println!(
        "(c) mla_naive_paged_selected: max abs diff {worst:.3e} ({worst_quanta:.2} bf16 quanta); \
         every (row, head) sits at least {apart:.3e} from the DENSE reading"
    );
    assert!(
        worst_quanta <= 2.0,
        "the selected reading drifted {worst_quanta:.2} quanta from the softmax over its own keys"
    );
    assert!(
        apart > 20.0 * worst,
        "some (row, head)'s selected reading is indistinguishable from the dense one — \
         the index row may not be read at all"
    );
}

// ---------------------------------------------------------------------------
// (d) The absorbed chain, and the V-block base.
// ---------------------------------------------------------------------------

/// The candidate bases for `kv_b`'s value planes, in whole `nope`-blocks past
/// head `h`'s own base. Index 1 is what the shader spells.
const CANDIDATES: [(&str, usize); 3] = [
    ("kv_b + 0            (the key-up block read as the value-up block)", 0),
    ("kv_b + nope*rank    (the CUDA byte add, read as bytes — THE SHADER'S)", 1),
    ("kv_b + 2*nope*rank  (the CUDA byte add, misread as elements)", 2),
];

/// (d), **THE PRIZE**: `absorb_q` → `mla_naive_paged` → `absorb_out`, all
/// three on the device, against a CPU reference of the UNABSORBED attention —
/// keys and values materialized per key through `W_UK`/`W_UV`, no latent-space
/// shortcut. The absorbed identity is the claim; the V-block base is what the
/// claim is sensitive to, so all three candidate bases are measured and
/// printed rather than one being assumed.
#[test]
fn the_absorbed_pair_is_the_unabsorbed_attention() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the absorbed chain") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x51ee_d0a1);
    let pool = Pool::reserve(&device);
    fill_pool(&device, &pipelines, &handles, &pool, &fx);

    let rows = ROWS.len();

    // ── absorb_q, on the device ──────────────────────────────────────────
    let q_nope = staged(&device, &encode(&fx.q_nope));
    let kv_b = staged(&device, &encode(&fx.kv_b));
    let q_lat = Buffer::zeroed(&device, (rows * HEADS * RANK * 2) as u64).expect("the latent q reserves");
    let q_nope_h = bind_whole(&handles, &q_nope, "the unabsorbed q");
    let kv_b_h = bind_whole(&handles, &kv_b, "the kv_b weight");
    let q_lat_h = bind_whole(&handles, &q_lat, "the absorbed q");
    let weight_rows = (HEADS * (NOPE + VDIM) + NOPE) as u32;

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::mla::absorb_q(
            &sink,
            Tensor::new(q_nope_h, rows as u32, (HEADS * NOPE) as u32, Dtype::Bf16),
            Tensor::new(kv_b_h, weight_rows, RANK as u32, Dtype::Bf16),
            HEADS as u32,
            RANK as u32,
            NOPE as u32,
            VDIM as u32,
            Tensor::new(q_lat_h, rows as u32, (HEADS * RANK) as u32, Dtype::Bf16),
        )
        .expect("the q absorb encodes");
    }
    frame.commit().expect("the q absorb completes");

    // The absorbed q, as the device produced it — and the arithmetic check on
    // `absorb_q` itself, which no other gate makes.
    let device_q_lat = decode(&read_back(&q_lat, rows * HEADS * RANK * 2));
    let mut q_worst = 0.0f32;
    for row in 0..rows {
        for h in 0..HEADS {
            let want = fx.absorbed_q(row, h);
            let base = (row * HEADS + h) * RANK;
            for (i, w) in want.iter().enumerate() {
                q_worst = q_worst.max((device_q_lat[base + i] - w).abs() / quantum(w.abs().max(0.05)));
            }
        }
    }
    assert!(q_worst <= 2.0, "mla_absorb_q drifted {q_worst:.2} quanta");

    // ── the flash reader over the device's own absorbed q ────────────────
    let latent = fire_flash(&device, &pipelines, &handles, &pool, &fx, &device_q_lat, None);

    // ── absorb_out, on the device ────────────────────────────────────────
    let latent_buf = staged(&device, &encode(&latent));
    let o_buf = Buffer::zeroed(&device, (rows * HEADS * VDIM * 2) as u64).expect("the output reserves");
    let latent_h = bind_whole(&handles, &latent_buf, "the latent reading");
    let kv_b2_h = bind_whole(&handles, &kv_b, "the kv_b weight, again");
    let o_h = bind_whole(&handles, &o_buf, "the value-space output");

    let frame = device.frame().expect("a second command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::mla::absorb_out(
            &sink,
            Tensor::new(latent_h, rows as u32, (HEADS * RANK) as u32, Dtype::Bf16),
            Tensor::new(kv_b2_h, weight_rows, RANK as u32, Dtype::Bf16),
            HEADS as u32,
            RANK as u32,
            VDIM as u32,
            NOPE as u32,
            Tensor::new(o_h, rows as u32, (HEADS * VDIM) as u32, Dtype::Bf16),
        )
        .expect("the output absorb encodes");
    }
    frame.commit().expect("the output absorb completes");
    let got = decode(&read_back(&o_buf, rows * HEADS * VDIM * 2));

    // ── the candidate sweep ──────────────────────────────────────────────
    println!("(d) absorb_q -> mla_naive_paged -> absorb_out vs the UNABSORBED attention:");
    let mut verdicts = Vec::new();
    for (label, blocks) in CANDIDATES {
        let mut worst = 0.0f32;
        let mut worst_quanta = 0.0f32;
        for row in 0..rows {
            for h in 0..HEADS {
                let want = cpu_unabsorbed(&fx, row, h, &dense_keys(row), blocks);
                let base = (row * HEADS + h) * VDIM;
                for (i, w) in want.iter().enumerate() {
                    let d = (got[base + i] - w).abs();
                    worst = worst.max(d);
                    worst_quanta = worst_quanta.max(d / quantum(w.abs().max(0.05)));
                }
            }
        }
        println!("    {label}  max abs diff {worst:.3e} ({worst_quanta:.1} bf16 quanta)");
        verdicts.push((label, blocks, worst, worst_quanta));
    }

    let inside: Vec<usize> = verdicts
        .iter()
        .filter(|(_, _, _, q)| *q <= 6.0)
        .map(|(_, b, _, _)| *b)
        .collect();
    assert_eq!(
        inside,
        vec![1],
        "exactly one candidate V-block base may reconcile with the unabsorbed \
         attention, and it must be `kv_b + nope*rank`: {verdicts:?}"
    );
    let chosen = verdicts[1].2;
    for (label, blocks, worst, _) in &verdicts {
        if *blocks != 1 {
            assert!(
                *worst > 20.0 * chosen,
                "`{label}` is not decisively wrong ({worst:.3e} against {chosen:.3e}) — \
                 the fixture does not separate the candidates"
            );
        }
    }
    println!(
        "    => the value-up block begins `nope*rank` ELEMENTS past head h's base; \
         the CUDA `2*nope*rank` is a BYTE add and the `2` is sizeof(bf16)"
    );
    println!("(d) mla_absorb_q max drift {q_worst:.2} quanta; the chain matches at {chosen:.3e}");
}

// ---------------------------------------------------------------------------
// The real geometry.
// ---------------------------------------------------------------------------

/// dsv4/glm5's own numbers, once: H=64, CKV=512, KPE=64. The small fixture
/// above is shaped to separate hypotheses; this one asks only whether the
/// launches this plane will really make — a 512-lane latent strip, 64 heads —
/// hold their arithmetic at all.
#[test]
fn the_absorbed_pair_holds_at_the_deepseek_geometry() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the deepseek geometry") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    const H: usize = 64;
    const CKV: usize = 512;
    const PE: usize = 64;
    const NO: usize = 128;
    const VD: usize = 128;
    const KEYS: usize = 6;
    const SLOTS: usize = 8;
    const TOKENS: usize = 2;

    let mut rng = Lcg(0xd504_9147);
    let ckv = rng.plane(KEYS * CKV);
    let kpe = rng.plane(KEYS * PE);
    let q_nope = rng.plane(TOKENS * H * NO);
    let q_pe = rng.plane(TOKENS * H * PE);
    let kv_b = rng.plane(H * (NO + VD) * CKV);

    // One request, two pages, the second partial — and page 1 before page 0 in
    // the table, so the page walk is still not the identity.
    let page_indices: [u32; 2] = [1, 0];
    let page_indptr: [u32; 2] = [0, 2];
    let positions: [i32; TOKENS] = [5, 2];
    let req_of: [i32; TOKENS] = [0, 0];
    let page = 4usize;
    let slot_of = |j: usize| page_indices[j / page] as usize * page + j % page;

    // The pool, staged directly: the appender has its own gate above.
    let mut ckv_pages = vec![0u8; SLOTS * CKV * 2];
    let mut kpe_pages = vec![0u8; SLOTS * PE * 2];
    for j in 0..KEYS {
        let s = slot_of(j);
        ckv_pages[s * CKV * 2..(s + 1) * CKV * 2]
            .copy_from_slice(&encode(&ckv[j * CKV..(j + 1) * CKV]));
        kpe_pages[s * PE * 2..(s + 1) * PE * 2]
            .copy_from_slice(&encode(&kpe[j * PE..(j + 1) * PE]));
    }

    let ckv_buf = staged(&device, &ckv_pages);
    let kpe_buf = staged(&device, &kpe_pages);
    let idx_buf = staged(&device, as_bytes_u32(&page_indices));
    let ptr_buf = staged(&device, as_bytes_u32(&page_indptr));
    let pos_buf = staged(&device, as_bytes_i32(&positions));
    let req_buf = staged(&device, as_bytes_i32(&req_of));
    let qn_buf = staged(&device, &encode(&q_nope));
    let qp_buf = staged(&device, &encode(&q_pe));
    let w_buf = staged(&device, &encode(&kv_b));
    let ql_buf = Buffer::zeroed(&device, (TOKENS * H * CKV * 2) as u64).expect("q_latent reserves");
    let lat_buf = Buffer::zeroed(&device, (TOKENS * H * CKV * 2) as u64).expect("the reading reserves");
    let o_buf = Buffer::zeroed(&device, (TOKENS * H * VD * 2) as u64).expect("the output reserves");

    let pool = KvPool {
        keys: Tensor::new(bind_whole(&handles, &ckv_buf, "ckv"), SLOTS as u32, CKV as u32, Dtype::Bf16),
        values: Tensor::new(bind_whole(&handles, &kpe_buf, "kpe"), SLOTS as u32, PE as u32, Dtype::Bf16),
        page_indices: Tensor::new(bind_whole(&handles, &idx_buf, "pages"), 2, 1, Dtype::U32),
        page_indptr: Tensor::new(bind_whole(&handles, &ptr_buf, "spans"), 2, 1, Dtype::U32),
        page_size: page as i32,
        seq_stride: CKV as u64,
        head_stride: CKV as u64,
    };
    let qn_t = Tensor::new(bind_whole(&handles, &qn_buf, "q_nope"), TOKENS as u32, (H * NO) as u32, Dtype::Bf16);
    let qp_t = Tensor::new(bind_whole(&handles, &qp_buf, "q_pe"), TOKENS as u32, (H * PE) as u32, Dtype::Bf16);
    let w_t = Tensor::new(bind_whole(&handles, &w_buf, "kv_b"), (H * (NO + VD)) as u32, CKV as u32, Dtype::Bf16);
    let ql_t = Tensor::new(bind_whole(&handles, &ql_buf, "q_latent"), TOKENS as u32, (H * CKV) as u32, Dtype::Bf16);
    let lat_t = Tensor::new(bind_whole(&handles, &lat_buf, "reading"), TOKENS as u32, (H * CKV) as u32, Dtype::Bf16);
    let o_t = Tensor::new(bind_whole(&handles, &o_buf, "o"), TOKENS as u32, (H * VD) as u32, Dtype::Bf16);
    let pos_t = Tensor::new(bind_whole(&handles, &pos_buf, "positions"), TOKENS as u32, 1, Dtype::I32);
    let req_t = Tensor::new(bind_whole(&handles, &req_buf, "requests"), TOKENS as u32, 1, Dtype::I32);
    let scale = 1.0 / ((CKV + PE) as f32).sqrt();

    // The whole chain in one frame is three dependent dispatches; each gets its
    // own command buffer, so the ordering rests on the commit rather than on
    // this file's reading of Metal's hazard tracking.
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::mla::absorb_q(&sink, qn_t, w_t, H as u32, CKV as u32, NO as u32, VD as u32, ql_t)
            .expect("the q absorb encodes");
    }
    frame.commit().expect("the q absorb completes");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::mla::attention_decode(
            &sink, ql_t, qp_t, &pool, pos_t, req_t, H as u32, CKV as u32, scale, lat_t,
        )
        .expect("the reader encodes");
    }
    frame.commit().expect("the reader completes");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::attn::mla::absorb_out(
            &sink, lat_t, w_t, H as u32, CKV as u32, VD as u32, NO as u32, o_t,
        )
        .expect("the output absorb encodes");
    }
    frame.commit().expect("the output absorb completes");
    let got = decode(&read_back(&o_buf, TOKENS * H * VD * 2));

    // The unabsorbed reference, one head at a time.
    let w_uk = |h: usize, k: usize| {
        let b = (h * (NO + VD) + k) * CKV;
        &kv_b[b..b + CKV]
    };
    let w_uv = |h: usize, k: usize| {
        let b = (h * (NO + VD) + NO + k) * CKV;
        &kv_b[b..b + CKV]
    };
    let mut worst = 0.0f32;
    let mut worst_quanta = 0.0f32;
    for t in 0..TOKENS {
        let keys: Vec<usize> = (0..=positions[t] as usize).collect();
        for h in 0..H {
            let qn = &q_nope[(t * H + h) * NO..(t * H + h + 1) * NO];
            let qp = &q_pe[(t * H + h) * PE..(t * H + h + 1) * PE];
            let mut scores = Vec::with_capacity(keys.len());
            let mut values = Vec::with_capacity(keys.len());
            for &j in &keys {
                let k = &ckv[j * CKV..(j + 1) * CKV];
                let k_nope: Vec<f32> = (0..NO).map(|n| dot(w_uk(h, n), k)).collect();
                scores.push((dot(qn, &k_nope) + dot(qp, &kpe[j * PE..(j + 1) * PE])) * scale);
                values.push((0..VD).map(|v| dot(w_uv(h, v), k)).collect());
            }
            let want = reading(&scores, &values, VD);
            let base = (t * H + h) * VD;
            for (i, w) in want.iter().enumerate() {
                let d = (got[base + i] - w).abs();
                worst = worst.max(d);
                worst_quanta = worst_quanta.max(d / quantum(w.abs().max(0.05)));
            }
        }
    }
    println!(
        "(dsv4) H={H} CKV={CKV} KPE={PE} nope={NO} v={VD}: the absorbed chain matches the \
         unabsorbed attention at max abs diff {worst:.3e} ({worst_quanta:.1} bf16 quanta)"
    );
    // **A WIDER BAND THAN THE SMALL FIXTURE'S, AND IT IS NOT A WEAKER CLAIM.**
    // The chain stores bf16 twice between two contractions that are 512 deep
    // here rather than 64, and the value-space answer is a heavily cancelling
    // sum — so the accumulated rounding lands a few percent of the answer,
    // measured at ~7.5 quanta. The band is set with headroom above what this
    // machine measures because the drift is a property of the arithmetic, not
    // of a bug, and nothing about the V-block packing rests on it: gate (d)'s
    // small fixture separates the candidate bases by six hundred times this.
    assert!(
        worst_quanta <= 16.0,
        "the absorbed chain drifted {worst_quanta:.1} quanta at the deepseek geometry"
    );
}

// ---------------------------------------------------------------------------
// The one launch (b), (c) and (d) share.
// ---------------------------------------------------------------------------

/// Fires the flash reader over the fixture's pool with the given absorbed
/// query, dense when `selection` is `None` and selected when it is not, and
/// reads the `[rows, HEADS, RANK]` latent answer back.
fn fire_flash(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    pool: &Pool,
    fx: &Fixture,
    q_lat: &[f32],
    selection: Option<&[i32]>,
) -> Vec<f32> {
    let rows = ROWS.len();
    let positions: Vec<i32> = ROWS.iter().map(|(_, p)| *p).collect();
    let requests: Vec<i32> = ROWS.iter().map(|(r, _)| *r as i32).collect();

    let q_buf = staged(device, &encode(q_lat));
    let qp_buf = staged(device, &encode(&fx.q_pe));
    let pos_buf = staged(device, as_bytes_i32(&positions));
    let req_buf = staged(device, as_bytes_i32(&requests));
    let out = Buffer::zeroed(device, (rows * HEADS * RANK * 2) as u64).expect("the reading reserves");
    let sel_buf = selection.map(|s| staged(device, as_bytes_i32(s)));

    let q_t = Tensor::new(bind_whole(handles, &q_buf, "q_latent"), rows as u32, (HEADS * RANK) as u32, Dtype::Bf16);
    let qp_t = Tensor::new(bind_whole(handles, &qp_buf, "q_pe"), rows as u32, (HEADS * KPE) as u32, Dtype::Bf16);
    let pos_t = Tensor::new(bind_whole(handles, &pos_buf, "positions"), rows as u32, 1, Dtype::I32);
    let req_t = Tensor::new(bind_whole(handles, &req_buf, "requests"), rows as u32, 1, Dtype::I32);
    let out_t = Tensor::new(bind_whole(handles, &out, "the reading"), rows as u32, (HEADS * RANK) as u32, Dtype::Bf16);
    let sel_t = sel_buf.as_ref().map(|b| {
        Tensor::new(bind_whole(handles, b, "the selection"), rows as u32, TOPK as u32, Dtype::I32)
    });
    let view = pool.view(handles);

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        match sel_t {
            None => kernels_metal::attn::mla::attention_decode(
                &sink, q_t, qp_t, &view, pos_t, req_t, HEADS as u32, RANK as u32, sm_scale(), out_t,
            ),
            Some(sel) => kernels_metal::attn::mla::attention_decode_selected(
                &sink, q_t, qp_t, sel, &view, pos_t, req_t, HEADS as u32, RANK as u32, sm_scale(), out_t,
            ),
        }
        .expect("the latent reader encodes");
    }
    frame.commit().expect("the reader completes");
    decode(&read_back(&out, rows * HEADS * RANK * 2))
}

// ---------------------------------------------------------------------------
// Plumbing — `tower_kernels_on_device`'s, unchanged where it applies.
// ---------------------------------------------------------------------------

/// A deterministic stream, so a failure here is reproducible from the seed
/// printed in the fixture's construction rather than from a captured file.
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

    /// `n` values in `[-0.25, 0.25)`, **already through bf16** — the reference
    /// and the device must be reading the same numbers, so the truncation the
    /// staging performs happens here too.
    fn plane(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32_of(bf16_bits(self.next_f32()))).collect()
    }
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// f32 → the two bytes of its bf16 truncation, little-endian.
fn bf16(v: f32) -> [u8; 2] {
    bf16_bits(v).to_le_bytes()
}

/// bf16 bits as the f32 they read back as.
fn f32_of(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn encode(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| bf16(*v)).collect()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32_of(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// The bf16 quantum at `v`: eight significant bits below the binade.
fn quantum(v: f32) -> f32 {
    if v == 0.0 {
        return f32::MIN_POSITIVE;
    }
    v.abs().log2().floor().exp2() / 128.0
}

/// A reservation holding `bytes`.
fn staged(device: &Context, bytes: &[u8]) -> Buffer {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("the reservation lands");
    buffer.write(0, bytes).expect("the bytes land");
    buffer
}

/// A handle over the whole of a reservation.
fn bind_whole(handles: &Handles, buffer: &Buffer, what: &str) -> u32 {
    handles
        .bind(buffer, 0, buffer.bytes())
        .unwrap_or_else(|fault| panic!("{what} binds: {fault}"))
}

/// The bytes of a reservation, read back.
fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut got = vec![0u8; bytes];
    buffer.read(0, &mut got).expect("the answer reads back");
    got
}

/// An `i32` slice as the bytes the shell would stage.
fn as_bytes_i32(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` has no padding and no invalid bit patterns, and the
    // slice's lifetime is the borrow's.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// A `u32` slice, likewise.
fn as_bytes_u32(values: &[u32]) -> &[u8] {
    // SAFETY: as `as_bytes_i32`.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
