//! The fa2 attention points as CLAIMED BODIES, and the door they reach their
//! staging through.
//!
//! `attention.{decode,decode_lse,prefill,prefill_lse,masked,kv_append}` were
//! claim-only until R4b for one reason spelled six ways: the launcher wants a
//! SCHEDULE, or a host CSR mirror, or a mask, or the fire's write origin, and
//! a statement carries none of those. What closed it is not a slot — the
//! declarations are unchanged — it is a door on the plane's own context:
//! `Ctx::raised::<R>()` asks the executor for the key `R::KEY` declares, and
//! `driver-cuda`'s `FireViews` answers.
//!
//! # What each test can see
//!
//! * **The door delivers, and delivers the same thing.** A decode fired
//!   through the claim body and a decode fired through the routine directly,
//!   against ONE plan and ONE pool, must agree BIT FOR BIT — same output, same
//!   log-sum-exp. Anything the body got wrong on the way (a plan pointer that
//!   is not the staged one, a window that is not `-1`, a soft cap that is not
//!   zero, an operand re-marked at the wrong element) moves a bit.
//! * **The door refuses by NAME.** A context that stages nothing is what every
//!   hand-written call in this crate builds, and a body on one must say which
//!   key it wanted rather than dereference a null. `"fa2.decode"` is the whole
//!   message.
//! * **The two accounts of the head width are checked.** The statement states
//!   the width its rectangles were cut at; the schedule was planned at the
//!   width the executor asked for. They agree by construction in a real fire
//!   and a mutation here proves the body notices when they do not.
//! * **`attention.kv_append` lands where the CSR says.** Two planes, not one
//!   — which is the difference from `kv_append_shared.rs`, and the reason it
//!   is a separate test: an append that wrote the key plane into both halves
//!   would pass that file's assertions and fail these.
//!
//! The toy pool is `kv_append_shared.rs`'s, at a geometry the fa2 decode
//! kernel is instantiated for: `head_dim = 64`, which is the narrowest arm
//! `decode_root` carries.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Attention;
use kernels::raises::{Answered, Struct};
use kernels::routine::{Cache, In, Out};
use kernels::Refusal;
use kernels_cuda::attn::fa2::plan::{DecodePlanCache, Planned};
use kernels_cuda::jit::abi::{bf16, Tensor};
use kernels_cuda::jit::Ctx;
use kernels_cuda::views::{KvCache, PagedKvView};

/// The device scratch is a process-global named-slab arena sized for one fire
/// at a time, which a driver's stream serialization guarantees and a test
/// harness's thread pool does not. `gdn_chunk_prefill.rs`'s lock, verbatim
/// and for its reason.
static FIRE: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── the device, or a skip ────────────────────────────────────────────────

fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

fn device_or_skip(what: &str) -> bool {
    let Some(count) = quietly(|| {
        let mut n: i32 = 0;
        let code = unsafe { rt::cudaGetDeviceCount(&raw mut n) };
        (code == rt::cudaError::cudaSuccess).then_some(n)
    }) else {
        eprintln!("skipping {what}: no CUDA runtime library on this machine");
        return false;
    };
    match count {
        Some(n) if n > 0 => {}
        _ => {
            eprintln!("skipping {what}: no CUDA device this build can drive");
            return false;
        }
    }
    assert_eq!(
        unsafe { rt::cudaSetDevice(0) },
        rt::cudaError::cudaSuccess,
        "a device is present but cudaSetDevice(0) failed"
    );
    assert_eq!(
        unsafe { rt::cudaFree(core::ptr::null_mut()) },
        rt::cudaError::cudaSuccess,
        "a device is present but the primary context would not come up"
    );
    true
}

// ── device memory, freed when the run ends ───────────────────────────────

struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

impl Slab {
    fn of(bytes: &[u8]) -> Slab {
        let mut ptr: *mut c_void = core::ptr::null_mut();
        assert_eq!(
            unsafe { rt::cudaMalloc(&raw mut ptr, bytes.len().max(1)) },
            rt::cudaError::cudaSuccess,
            "cudaMalloc({})",
            bytes.len()
        );
        let slab = Slab {
            ptr,
            bytes: bytes.len().max(1),
        };
        if !bytes.is_empty() {
            assert_eq!(
                unsafe {
                    rt::cudaMemcpy(
                        slab.ptr,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    )
                },
                rt::cudaError::cudaSuccess,
                "host to device"
            );
        }
        slab
    }

    fn zeroed(bytes: usize) -> Slab {
        Slab::of(&vec![0u8; bytes.max(1)])
    }

    fn read(&self, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        assert_eq!(
            unsafe { rt::cudaDeviceSynchronize() },
            rt::cudaError::cudaSuccess,
            "device synchronize"
        );
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    self.ptr,
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            },
            rt::cudaError::cudaSuccess,
            "device to host"
        );
        out
    }

    fn read_u16(&self, elems: usize) -> Vec<u16> {
        self.read(elems * 2)
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    fn read_f32(&self, elems: usize) -> Vec<f32> {
        self.read(elems * 4)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { rt::cudaFree(self.ptr) };
        }
    }
}

// ── elements ─────────────────────────────────────────────────────────────

/// `__float2bfloat16`: round to nearest, ties to even.
fn narrow(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// xorshift64*, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / 8_388_608.0 - 1.0
    }

    fn bf16(&mut self, scale: f32) -> u16 {
        narrow(self.next() * scale)
    }
}

// ── the toy pool ─────────────────────────────────────────────────────────

/// One dense pool and the fire that reads and writes it.
///
/// The awkwardness is `kv_append_shared.rs`'s and is deliberate: scattered
/// page indices, two requests owning different page counts, a partly-full
/// last page each, and one INVALID row. `head_dim` is 64 because that is the
/// narrowest width `fa2::decode_root` is instantiated at, and the decode test
/// below has to reach a real kernel.
struct Toy {
    kv_heads: i32,
    q_heads: i32,
    head_dim: i32,
    page_size: i32,
    pool_pages: i32,
    qo_indptr: Vec<u32>,
    page_indptr: Vec<u32>,
    page_indices: Vec<u32>,
    last_page_lens: Vec<u32>,
    row_valid: Vec<u8>,
}

impl Toy {
    /// The APPEND's geometry: many rows per request, one of them rejected.
    fn appending() -> Toy {
        Toy {
            kv_heads: 2,
            q_heads: 2,
            head_dim: 64,
            page_size: 4,
            pool_pages: 8,
            // request 0: 2 new rows; request 1: 3 new rows.
            qo_indptr: vec![0, 2, 5],
            page_indptr: vec![0, 2, 3],
            // Scattered, and NOT in slab order.
            page_indices: vec![5, 2, 7],
            // request 0 holds 6 tokens (4 + 2), request 1 holds 3.
            last_page_lens: vec![2, 3],
            row_valid: vec![1, 1, 0, 1, 1],
        }
    }

    /// The DECODE's geometry: one query row per request, and a GQA group of
    /// four so the arm the plan picks is a real one.
    fn decoding() -> Toy {
        Toy {
            kv_heads: 2,
            q_heads: 8,
            head_dim: 64,
            page_size: 4,
            pool_pages: 8,
            qo_indptr: vec![0, 1, 2],
            page_indptr: vec![0, 2, 3],
            page_indices: vec![5, 2, 7],
            last_page_lens: vec![2, 3],
            row_valid: vec![1, 1],
        }
    }

    fn row(&self) -> i32 {
        self.kv_heads * self.head_dim
    }

    fn q_row(&self) -> i32 {
        self.q_heads * self.head_dim
    }

    fn rows(&self) -> i32 {
        *self.qo_indptr.last().expect("a CSR ends somewhere") as i32
    }

    fn requests(&self) -> i32 {
        self.qo_indptr.len() as i32 - 1
    }

    fn plane_elems(&self) -> usize {
        (self.pool_pages * self.page_size * self.row()) as usize
    }

    /// The page and offset row `t` writes, on the host.
    ///
    /// A transcription of `pie::attn::write_kv`'s prologue
    /// (`attn/kv_paged.cuh`) and of nothing else.
    fn destination(&self, t: i32) -> (i32, i32) {
        let r = (0..self.requests())
            .find(|r| t < self.qo_indptr[*r as usize + 1] as i32)
            .unwrap_or(self.requests() - 1) as usize;
        let qo_lo = self.qo_indptr[r] as i32;
        let qo_hi = self.qo_indptr[r + 1] as i32;
        let new_tokens = qo_hi - qo_lo;
        let pages_first = self.page_indptr[r] as i32;
        let pages_last = self.page_indptr[r + 1] as i32;
        let total_kv_after =
            (pages_last - pages_first - 1) * self.page_size + self.last_page_lens[r] as i32;
        let abs = total_kv_after - new_tokens + (t - qo_lo);
        (
            self.page_indices[(pages_first + abs / self.page_size) as usize] as i32,
            abs % self.page_size,
        )
    }
}

/// The pool's device planes and the view a statement names.
struct Pool {
    keys: Slab,
    values: Slab,
    _indices: Slab,
    _indptr: Slab,
    _lens: Slab,
    _qo: Slab,
    _valid: Slab,
    view: PagedKvView,
}

impl Pool {
    fn build(toy: &Toy, poison: (f32, f32)) -> Pool {
        // A POISON FILL, not zeros: a body that wrote nothing at all would
        // pass a zero-vs-zero comparison at every slot it was supposed to
        // skip, and the reference expects the poison to survive.
        let keys = Slab::of(&bytes_of_u16(&vec![narrow(poison.0); toy.plane_elems()]));
        let values = Slab::of(&bytes_of_u16(&vec![narrow(poison.1); toy.plane_elems()]));
        let indices = Slab::of(&bytes_of_u32(&toy.page_indices));
        let indptr = Slab::of(&bytes_of_u32(&toy.page_indptr));
        let lens = Slab::of(&bytes_of_u32(&toy.last_page_lens));
        let qo = Slab::of(&bytes_of_u32(&toy.qo_indptr));
        let valid = Slab::of(&toy.row_valid);
        let view = PagedKvView {
            keys: keys.ptr.cast(),
            values: values.ptr.cast(),
            bf16_keys: keys.ptr.cast(),
            bf16_values: values.ptr.cast(),
            page_indices: indices.ptr.cast(),
            page_indptr: indptr.ptr.cast(),
            last_page_lens: lens.ptr.cast(),
            key_scales: core::ptr::null(),
            value_scales: core::ptr::null(),
            write_page: core::ptr::null(),
            write_offset: core::ptr::null(),
            page_size: toy.page_size,
            // NHD, in ELEMENTS: a page is `[page_size, kv_heads, head_dim]`,
            // so a token step crosses every head and a head is `head_dim`.
            // THIS PAIR IS THE APPEND BODY'S ONLY SOURCE for the head split.
            seq_stride: i64::from(toy.kv_heads) * i64::from(toy.head_dim),
            head_stride: i64::from(toy.head_dim),
            layout: 0,
            storage_dtype: kernels_cuda::attn::KvDType::Bf16 as i32,
            scheme_byte: kernels_cuda::attn::KvScheme::Native as i32,
            native_bf16: true,
            has_envelopes: false,
            env_min: core::ptr::null(),
            env_max: core::ptr::null(),
            block_size: 0,
            max_pages_per_request: toy.page_indices.len() as i32,
            pages_in_batch: toy.page_indices.len() as i32,
            qo_indptr: qo.ptr.cast(),
            row_valid: valid.ptr.cast(),
            requests: toy.requests(),
        };
        Pool {
            keys,
            values,
            _indices: indices,
            _indptr: indptr,
            _lens: lens,
            _qo: qo,
            _valid: valid,
            view,
        }
    }

    fn cache(&self) -> Cache<Struct<KvCache>> {
        Cache {
            ptr: core::ptr::from_ref(&self.view),
        }
    }
}

// ── the staging this test answers, by key ────────────────────────────────

/// What an executor is, to a claim body: an answer for a `Raise`'s key.
///
/// `driver-cuda`'s `FireViews` is the real one and `baker-smoke`'s `Staged`
/// is the one-row one; this is the two-request one. Every field is a raw
/// pointer the test owns for the whole run.
struct Staged {
    decode_plan: *const c_void,
}

impl Answered for Staged {
    fn raised(&self, key: &'static str) -> Option<*const c_void> {
        match key {
            "fa2.decode" => Some(self.decode_plan),
            _ => None,
        }
    }
}

/// The workspaces the schedule is stamped into. `baker-smoke`'s figures.
const ATTN_FLOAT_BYTES: usize = 32 << 20;
const ATTN_INT_BYTES: usize = 16 << 20;

/// One decode schedule, planned exactly the way a driver raises one.
///
/// The workspaces are stamped BEFORE the plan, `enable_cuda_graph` is on and
/// the variant is FULL with `window_left = -1` — the three things
/// `driver-cuda/src/fire/launch.rs::raise_attn_plans` does and that
/// `baker-smoke/src/smoke.rs::plan_decode` records the reasons for. A plan
/// raised any other way is not the plan a fire would have.
fn plan(toy: &Toy, head_dim: i32, float_ws: &Slab, int_ws: &Slab) -> Box<DecodePlanCache> {
    let mut cache = Box::new(DecodePlanCache::new());
    cache.int_workspace = int_ws.ptr;
    cache.float_workspace = float_ws.ptr;
    cache.set_int_base(0);
    let device = kernels_cuda::attn::fa2::plan::plan_device();
    let max_grid = kernels_cuda::attn::fa2::plan::decode_max_grid_size(
        head_dim,
        toy.q_heads,
        toy.kv_heads,
    );
    let planned = kernels_cuda::attn::fa2::plan::plan_decode(
        &mut cache,
        &toy.page_indptr,
        toy.requests(),
        toy.q_heads,
        toy.kv_heads,
        head_dim,
        toy.page_size,
        kernels_cuda::attn::plan::Workspace::new(float_ws.bytes, int_ws.bytes),
        &device,
        max_grid,
        true,
        true,
        false,
        -1,
    );
    match planned {
        Planned::Full | Planned::StaticNonsplit => cache,
        Planned::Declined(why) => panic!("the fa2 decode planner declined: {why}"),
    }
}

// ── the append ───────────────────────────────────────────────────────────

#[test]
fn the_append_lands_both_planes_where_the_csr_says() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.kv_append") {
        return;
    }
    let toy = Toy::appending();
    let pool = Pool::build(&toy, (-7.5, -3.25));
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let n = (toy.rows() * toy.row()) as usize;
    // TWO DIFFERENT PLANES, which is the whole difference from
    // `kv_append_shared.rs`: a body that passed `k` for both operands would
    // pass that file's assertions and fail this one at every value element.
    let k: Vec<u16> = (0..n).map(|_| rng.bf16(2.0)).collect();
    let v: Vec<u16> = (0..n).map(|_| rng.bf16(3.0)).collect();

    let d_k = Slab::of(&bytes_of_u16(&k));
    let d_v = Slab::of(&bytes_of_u16(&v));
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let mark = |s: &Slab| In::<Tensor<bf16>> {
        ptr: s.ptr.cast(),
        rows: toy.rows(),
        width: toy.row(),
    };
    Attention::kv_append::<bf16>(&ctx, mark(&d_k), mark(&d_v), pool.cache())
        .expect("the claimed `attention.kv_append` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the append did not complete"
    );

    // The reference: each plane's poison, overwritten only where a VALID
    // row's destination says, and from that plane's OWN source.
    let mut want_k = vec![narrow(-7.5); toy.plane_elems()];
    let mut want_v = vec![narrow(-3.25); toy.plane_elems()];
    for t in 0..toy.rows() {
        if toy.row_valid[t as usize] == 0 {
            continue;
        }
        let (page, off) = toy.destination(t);
        let dst = ((page * toy.page_size + off) * toy.row()) as usize;
        for i in 0..toy.row() as usize {
            want_k[dst + i] = k[t as usize * toy.row() as usize + i];
            want_v[dst + i] = v[t as usize * toy.row() as usize + i];
        }
    }

    let elems = toy.plane_elems();
    let got_k = pool.keys.read_u16(elems);
    let got_v = pool.values.read_u16(elems);
    let bad_k = (0..elems).filter(|i| got_k[*i] != want_k[*i]).count();
    let bad_v = (0..elems).filter(|i| got_v[*i] != want_v[*i]).count();
    eprintln!(
        "attention.kv_append: keys {}/{elems} exact, values {}/{elems} exact",
        elems - bad_k,
        elems - bad_v
    );
    // A COPY IS EXACT OR IT IS WRONG. There is no arithmetic in this kernel
    // to give a tolerance to.
    assert_eq!(bad_k, 0, "{bad_k} element(s) landed wrong in the key plane");
    assert_eq!(
        bad_v, 0,
        "{bad_v} element(s) landed wrong in the value plane"
    );
}

/// THE MUTATION: the two planes are not interchangeable.
///
/// The body hands `k` and `v` to a kernel that writes two DISTINCT
/// destinations, and the failure it has to rule out is an append that reads
/// one source twice — which is exactly what `kv_append_shared` legitimately
/// does, one method away. This asserts the value plane holds the VALUE rows
/// and not the key ones, which is the assertion the shared test cannot make.
#[test]
fn the_value_plane_is_not_the_key_plane() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.kv_append") {
        return;
    }
    let toy = Toy::appending();
    let pool = Pool::build(&toy, (0.0, 0.0));
    let n = (toy.rows() * toy.row()) as usize;
    // Two constants, so the check is a single number per plane.
    let k = vec![narrow(1.5); n];
    let v = vec![narrow(-2.25); n];
    let d_k = Slab::of(&bytes_of_u16(&k));
    let d_v = Slab::of(&bytes_of_u16(&v));
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let mark = |s: &Slab| In::<Tensor<bf16>> {
        ptr: s.ptr.cast(),
        rows: toy.rows(),
        width: toy.row(),
    };
    Attention::kv_append::<bf16>(&ctx, mark(&d_k), mark(&d_v), pool.cache())
        .expect("the claimed `attention.kv_append` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the append did not complete"
    );
    let (page, off) = toy.destination(0);
    let at = ((page * toy.page_size + off) * toy.row()) as usize;
    let got_k = pool.keys.read_u16(toy.plane_elems());
    let got_v = pool.values.read_u16(toy.plane_elems());
    assert_eq!(got_k[at], narrow(1.5), "the key plane holds the key row");
    assert_eq!(
        got_v[at],
        narrow(-2.25),
        "the value plane holds the KEY row, so the body read one source twice"
    );
}

// ── the decode, through the door and around it ───────────────────────────

/// Fire `attention.decode` twice — once through the claim body, which asks
/// the executor for `"fa2.decode"`, and once through the routine with the
/// same plan handed to it directly — and demand the two answers be identical
/// to the bit.
///
/// THIS IS THE A/B AGAINST THE PRE-CHANGE PATH. The second call is literally
/// what `driver-cuda/src/baker/staging.rs`'s deleted arm did: the same
/// routine, the same plan pointer, `window_left = -1`, `logits_soft_cap = 0`,
/// the same `sm_scale`. If the door hands the body anything else — a stale
/// plan, a different window convention, an operand re-marked at the wrong
/// element — the two rows differ.
#[test]
fn the_body_and_the_routine_answer_the_same_decode() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.decode") {
        return;
    }
    let toy = Toy::decoding();
    let pool = Pool::build(&toy, (0.0, 0.0));
    // Real keys and values in the pages, so the softmax has something to do.
    let mut rng = Rng(0x0bad_c0de_dead_beef);
    let filled: Vec<u16> = (0..toy.plane_elems()).map(|_| rng.bf16(1.0)).collect();
    let keys = Slab::of(&bytes_of_u16(&filled));
    let values: Vec<u16> = (0..toy.plane_elems()).map(|_| rng.bf16(1.0)).collect();
    let values = Slab::of(&bytes_of_u16(&values));
    let mut view = pool.view;
    view.keys = keys.ptr.cast();
    view.values = values.ptr.cast();
    view.bf16_keys = keys.ptr.cast();
    view.bf16_values = values.ptr.cast();
    let cache = Cache::<Struct<KvCache>> {
        ptr: core::ptr::from_ref(&view),
    };

    let float_ws = Slab::zeroed(ATTN_FLOAT_BYTES);
    let int_ws = Slab::zeroed(ATTN_INT_BYTES);
    let schedule = plan(&toy, toy.head_dim, &float_ws, &int_ws);

    let q_elems = (toy.rows() * toy.q_row()) as usize;
    let q: Vec<u16> = (0..q_elems).map(|_| rng.bf16(1.0)).collect();
    let d_q = Slab::of(&bytes_of_u16(&q));
    let o_body = Slab::zeroed(q_elems * 2);
    let o_routine = Slab::zeroed(q_elems * 2);
    let lse_body = Slab::zeroed((toy.rows() * toy.q_heads) as usize * 4);
    let lse_routine = Slab::zeroed((toy.rows() * toy.q_heads) as usize * 4);

    let q_mark = In::<Tensor<bf16>> {
        ptr: d_q.ptr.cast(),
        rows: toy.rows(),
        width: toy.q_row(),
    };
    let o_mark = |s: &Slab| Out::<Tensor<bf16>> {
        ptr: s.ptr.cast(),
        rows: toy.rows(),
        width: toy.q_row(),
    };
    let lse_mark = |s: &Slab| Out::<Tensor<f32>> {
        ptr: s.ptr.cast(),
        rows: toy.rows(),
        width: toy.q_heads,
    };
    let sm_scale = 1.0 / (toy.head_dim as f32).sqrt();

    // A: the claimed body, which reaches the schedule by KEY.
    let staged = Staged {
        decode_plan: (&raw const *schedule).cast::<c_void>(),
    };
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) }.with_raised(&staged);
    Attention::decode_lse::<bf16>(
        &ctx,
        q_mark,
        cache,
        0,
        toy.head_dim.unsigned_abs(),
        sm_scale,
        o_mark(&o_body),
        lse_mark(&lse_body),
    )
    .expect("the claimed `attention.decode_lse` body");

    // B: the routine, handed the same plan the way the deleted staging arm
    // handed it.
    let bare = unsafe { Ctx::on(core::ptr::null_mut()) };
    kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode(
        &bare,
        q_mark,
        In {
            ptr: (&raw const *schedule),
            rows: 0,
            width: 0,
        },
        o_mark(&o_routine),
        kernels::routine::Const::new(-1),
        kernels::routine::Const::new(0.0),
        kernels::routine::Const::new(sm_scale),
        cache.raised(),
        Some(lse_mark(&lse_routine)),
    )
    .expect("the routine the staging arm used to call");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the two decodes did not complete"
    );

    let a = o_body.read_u16(q_elems);
    let b = o_routine.read_u16(q_elems);
    let bad = (0..q_elems).filter(|i| a[*i] != b[*i]).count();
    eprintln!("attention.decode A/B: {}/{q_elems} bit-identical", q_elems - bad);
    assert_eq!(
        bad, 0,
        "{bad} output element(s) differ between the claimed body and the routine"
    );
    let heads = (toy.rows() * toy.q_heads) as usize;
    let la = lse_body.read_f32(heads);
    let lb = lse_routine.read_f32(heads);
    let bad_lse = (0..heads).filter(|i| la[*i].to_bits() != lb[*i].to_bits()).count();
    assert_eq!(bad_lse, 0, "{bad_lse} log-sum-exp element(s) differ");
    // AND THE ANSWER IS NOT NOTHING: a pair of zero rows would agree too.
    assert!(
        a.iter().any(|x| *x != 0),
        "both decodes wrote zeros; the A/B proves nothing"
    );
}

// ── the door itself ──────────────────────────────────────────────────────

/// A context that stages nothing refuses BY THE KEY, and does not fault.
///
/// Every hand-written call in this crate builds exactly such a context —
/// `Ctx::on(stream)` and no more — so this is the ordinary shape of a body
/// asked for staging that is not there, and the message has to name the thing
/// to build. `driver-cuda` answers `None` for `"mla.plan"` and the dsv4 slabs
/// today for the same reason, and prints the same word.
#[test]
fn a_context_that_stages_nothing_refuses_by_the_key() {
    let toy = Toy::decoding();
    let pool = Pool::build(&toy, (0.0, 0.0));
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let q = In::<Tensor<bf16>> {
        ptr: core::ptr::null(),
        rows: toy.rows(),
        width: toy.q_row(),
    };
    let o = Out::<Tensor<bf16>> {
        ptr: core::ptr::null_mut(),
        rows: toy.rows(),
        width: toy.q_row(),
    };
    let refused = Attention::decode::<bf16>(
        &ctx,
        q,
        pool.cache(),
        0,
        toy.head_dim.unsigned_abs(),
        1.0,
        o,
    );
    assert!(
        matches!(refused, Err(Refusal::Absent { what: "fa2.decode" })),
        "a body on an unstaged context must name the key it wanted, got {refused:?}"
    );
}

/// The two accounts of the head width, disagreeing.
///
/// The statement states the width its rectangles were cut at; the schedule
/// was planned at the width the executor asked for. In a real fire they agree
/// by construction (`Baked::attn_ask` reads the ask off the same statements),
/// so the only way to reach this is to stage a plan for another geometry —
/// which is what a driver raising one schedule for a stack that wants two
/// would do, silently, to half its launches.
#[test]
fn a_schedule_planned_at_another_width_is_refused() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.decode") {
        return;
    }
    let toy = Toy::decoding();
    let pool = Pool::build(&toy, (0.0, 0.0));
    let float_ws = Slab::zeroed(ATTN_FLOAT_BYTES);
    let int_ws = Slab::zeroed(ATTN_INT_BYTES);
    // 128, where the statement below states 64. Both are instantiated arms,
    // so nothing else refuses first.
    let schedule = plan(&toy, 128, &float_ws, &int_ws);
    let staged = Staged {
        decode_plan: (&raw const *schedule).cast::<c_void>(),
    };
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) }.with_raised(&staged);
    let q = In::<Tensor<bf16>> {
        ptr: core::ptr::null(),
        rows: toy.rows(),
        width: toy.q_row(),
    };
    let o = Out::<Tensor<bf16>> {
        ptr: core::ptr::null_mut(),
        rows: toy.rows(),
        width: toy.q_row(),
    };
    let refused = Attention::decode::<bf16>(&ctx, q, pool.cache(), 0, 64, 1.0, o);
    assert!(
        matches!(refused, Err(Refusal::Narrow { .. })),
        "a schedule planned at another head width must be refused, got {refused:?}"
    );
}
