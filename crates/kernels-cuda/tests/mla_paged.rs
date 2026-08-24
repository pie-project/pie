//! The latent-attention pair `W7` landed: `mla.kv_append`'s claimed body,
//! and the columned `mla.attention_decode` behind it.
//!
//! TWO DIFFERENT THINGS ARE UNDER TEST, and they are checked differently.
//!
//!  * `mla.kv_append` is claimed by a BODY, so there is no routine whose row
//!    a table test could check. What can be checked is where the bytes land,
//!    and the reference is the CSR arithmetic written out on the host — the
//!    same `pre_kv_len = total_kv_after - new_tokens` walk
//!    (`attn/mla_paged.cuh:128-158`) the kernel does. A body that read the
//!    pool row's planes in the wrong order, took the pitch off the wrong
//!    operand, or dropped the fire's row validity fails this and passes
//!    nothing else.
//!  * `mla.attention_decode` is CLAIM-ONLY: the point resolves through
//!    `attention_mla_decode_bf16`'s `canon` and the schedule is the plane's
//!    staging. What the routine owns is its OPERAND COLUMN — the `MlaLayer`
//!    it builds off the pool row, the rope width it divides out of `q_pe`,
//!    the request count it reads, the causal flag it picks — and every one
//!    of those is wrong-answer-shaped rather than crash-shaped. So it is
//!    fired against a host softmax over the pages the append just wrote:
//!    the two tests share one pool on purpose.
//!
//! # Which arm answers
//!
//! `dispatch_attention_mla_bf16` branches on compute capability: sm_100 and
//! up take the naive kernel, everything below takes flashinfer's FA2 MLA.
//! Both are exercised by the same test — whichever one this box has — and
//! the FA2 arm is the one that needs a plan, which is exactly why the point
//! is claim-only. `HEAD_DIM_CKV`/`HEAD_DIM_KPE` are compile-time constants
//! in that kernel (512 and 64), so "a small shape" here means few heads and
//! few keys, never a narrow latent.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Mla;
use kernels::raises::Struct;
use kernels::routine::{Cache, Const, In, Out};
use kernels_cuda::attn::plan::{self, Device, Workspace};
use kernels_cuda::attn::MlaPlan;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::jit::Ctx;
use kernels_cuda::views::{KvCache, PagedKvView};

/// The device scratch is a process-global named-slab arena sized for one
/// fire at a time, which the driver's stream serialization guarantees and a
/// test harness's thread pool does not. `gdn_chunk_prefill.rs`'s lock,
/// verbatim and for its reason.
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
        let slab = Slab { ptr };
        slab.upload(bytes);
        slab
    }

    fn zeroed(bytes: usize) -> Slab {
        Slab::of(&vec![0u8; bytes])
    }

    fn upload(&self, src: &[u8]) {
        if src.is_empty() {
            return;
        }
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    self.ptr,
                    src.as_ptr().cast(),
                    src.len(),
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            rt::cudaError::cudaSuccess,
            "host to device"
        );
    }

    fn read_u16(&self, elems: usize) -> Vec<u16> {
        let mut bytes = vec![0u8; elems * 2];
        assert_eq!(
            unsafe { rt::cudaDeviceSynchronize() },
            rt::cudaError::cudaSuccess,
            "device synchronize"
        );
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    bytes.as_mut_ptr().cast(),
                    self.ptr,
                    bytes.len(),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            },
            rt::cudaError::cudaSuccess,
            "device to host"
        );
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
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

fn wide(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

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

/// One latent pool and the fire that writes into it.
///
/// EVERY NUMBER HERE IS AWKWARD ON PURPOSE. The page indices are scattered
/// rather than `0..n`, the two requests own different page counts, the last
/// page of each is partly full, and one row is INVALID — a body that
/// ignored `row_valid`, walked the pages in request order, or assumed the
/// append starts at offset zero passes none of them.
struct Toy {
    kv_lora_rank: i32,
    rope_dim: i32,
    page_size: i32,
    pool_pages: i32,
    /// The fire's query CSR, `[requests + 1]`.
    qo_indptr: Vec<u32>,
    /// The page CSR, `[requests + 1]` into `page_indices`.
    page_indptr: Vec<u32>,
    /// The pages each request owns, in order.
    page_indices: Vec<u32>,
    /// How full each request's last page is AFTER this fire.
    last_page_lens: Vec<u32>,
    /// One byte per token row.
    row_valid: Vec<u8>,
}

impl Toy {
    fn small() -> Toy {
        Toy {
            kv_lora_rank: 8,
            rope_dim: 4,
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

    fn rows(&self) -> i32 {
        *self.qo_indptr.last().expect("a CSR ends somewhere") as i32
    }

    fn requests(&self) -> i32 {
        self.qo_indptr.len() as i32 - 1
    }

    /// The page and offset row `t` writes, on the host.
    ///
    /// A transcription of `pie::attn::mla_resolve_dst`
    /// (`attn/mla_paged.cuh:132-158`) and of nothing else.
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
    ckv: Slab,
    kpe: Slab,
    // Kept alive for the view's pointers.
    _indices: Slab,
    _indptr: Slab,
    _lens: Slab,
    _qo: Slab,
    _valid: Slab,
    view: PagedKvView,
}

impl Pool {
    fn build(toy: &Toy) -> Pool {
        let ckv_elems = (toy.pool_pages * toy.page_size * toy.kv_lora_rank) as usize;
        let kpe_elems = (toy.pool_pages * toy.page_size * toy.rope_dim) as usize;
        // A POISON FILL, not zeros: a body that wrote nothing at all would
        // pass a zero-vs-zero comparison at every slot it was supposed to
        // skip, and the reference below expects the poison to survive.
        let ckv = Slab::of(&bytes_of_u16(&vec![narrow(-7.5); ckv_elems]));
        let kpe = Slab::of(&bytes_of_u16(&vec![narrow(-7.5); kpe_elems]));
        let indices = Slab::of(&bytes_of_u32(&toy.page_indices));
        let indptr = Slab::of(&bytes_of_u32(&toy.page_indptr));
        let lens = Slab::of(&bytes_of_u32(&toy.last_page_lens));
        let qo = Slab::of(&bytes_of_u32(&toy.qo_indptr));
        let valid = Slab::of(&toy.row_valid);
        let view = PagedKvView {
            // THE LATENT POOL'S TWO PLANES, in the order
            // `pools/mla_cache.rs` allocates them: `ckv` then `kpe`.
            keys: ckv.ptr.cast(),
            values: kpe.ptr.cast(),
            bf16_keys: ckv.ptr.cast(),
            bf16_values: kpe.ptr.cast(),
            page_indices: indices.ptr.cast(),
            page_indptr: indptr.ptr.cast(),
            last_page_lens: lens.ptr.cast(),
            key_scales: core::ptr::null(),
            value_scales: core::ptr::null(),
            write_page: core::ptr::null(),
            write_offset: core::ptr::null(),
            page_size: toy.page_size,
            // The MLA planes have two different pitches and neither is the
            // view's; the statement's operands carry both. These two fields
            // are the dense cache's and no latent kernel reads them.
            seq_stride: 0,
            head_stride: 0,
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
            ckv,
            kpe,
            _indices: indices,
            _indptr: indptr,
            _lens: lens,
            _qo: qo,
            _valid: valid,
            view,
        }
    }

    fn row(&self) -> Cache<Struct<KvCache>> {
        Cache {
            ptr: core::ptr::from_ref(&self.view),
        }
    }
}

/// Fire `mla.kv_append` over `toy`, and answer the two planes it left.
fn append(toy: &Toy, pool: &Pool, kv_c: &[u16], k_pe: &[u16]) {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let d_kv_c = Slab::of(&bytes_of_u16(kv_c));
    let d_k_pe = Slab::of(&bytes_of_u16(k_pe));
    Mla::kv_append::<bf16>(
        &ctx,
        In {
            ptr: d_kv_c.ptr.cast(),
            rows: toy.rows(),
            width: toy.kv_lora_rank,
        },
        In {
            ptr: d_k_pe.ptr.cast(),
            rows: toy.rows(),
            width: toy.rope_dim,
        },
        pool.row(),
    )
    .expect("the claimed `mla.kv_append` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the append did not complete"
    );
}

#[test]
fn the_append_lands_where_the_csr_says() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("mla.kv_append") {
        return;
    }
    let toy = Toy::small();
    let pool = Pool::build(&toy);
    let mut rng = Rng(0x5eed_1234_9876_4321);
    let rows = toy.rows() as usize;
    let kv_c: Vec<u16> = (0..rows * toy.kv_lora_rank as usize)
        .map(|_| rng.bf16(2.0))
        .collect();
    let k_pe: Vec<u16> = (0..rows * toy.rope_dim as usize)
        .map(|_| rng.bf16(2.0))
        .collect();

    append(&toy, &pool, &kv_c, &k_pe);

    // The reference: the poison, overwritten only where a VALID row's
    // destination says.
    let ckv_elems = (toy.pool_pages * toy.page_size * toy.kv_lora_rank) as usize;
    let kpe_elems = (toy.pool_pages * toy.page_size * toy.rope_dim) as usize;
    let mut want_ckv = vec![narrow(-7.5); ckv_elems];
    let mut want_kpe = vec![narrow(-7.5); kpe_elems];
    for t in 0..toy.rows() {
        if toy.row_valid[t as usize] == 0 {
            continue;
        }
        let (page, off) = toy.destination(t);
        let ckv_dst = ((page * toy.page_size + off) * toy.kv_lora_rank) as usize;
        let kpe_dst = ((page * toy.page_size + off) * toy.rope_dim) as usize;
        for i in 0..toy.kv_lora_rank as usize {
            want_ckv[ckv_dst + i] = kv_c[t as usize * toy.kv_lora_rank as usize + i];
        }
        for i in 0..toy.rope_dim as usize {
            want_kpe[kpe_dst + i] = k_pe[t as usize * toy.rope_dim as usize + i];
        }
    }

    let got_ckv = pool.ckv.read_u16(ckv_elems);
    let got_kpe = pool.kpe.read_u16(kpe_elems);
    let bad_ckv = (0..ckv_elems).filter(|i| got_ckv[*i] != want_ckv[*i]).count();
    let bad_kpe = (0..kpe_elems).filter(|i| got_kpe[*i] != want_kpe[*i]).count();
    eprintln!(
        "mla.kv_append: ckv {}/{ckv_elems} exact, kpe {}/{kpe_elems} exact",
        ckv_elems - bad_ckv,
        kpe_elems - bad_kpe
    );
    // A COPY IS EXACT OR IT IS WRONG. There is no arithmetic in this kernel
    // to give a tolerance to.
    assert_eq!(bad_ckv, 0, "{bad_ckv} latent element(s) landed wrong");
    assert_eq!(bad_kpe, 0, "{bad_kpe} rotated element(s) landed wrong");
}

// ── the operand column, without a device ─────────────────────────────────

/// A pool row with nothing behind its pointers.
///
/// Every check below refuses BEFORE the dispatch touches the device, which
/// is what makes them checkable on a box whose only FA2 arm is the broken
/// one. The view is read on the HOST — that is the whole of what a `Cache`
/// mark carries — so a host struct with dangling device pointers is exactly
/// what these paths see.
fn blank_view() -> PagedKvView {
    PagedKvView {
        keys: core::ptr::null_mut(),
        values: core::ptr::null_mut(),
        bf16_keys: core::ptr::null_mut(),
        bf16_values: core::ptr::null_mut(),
        page_indices: core::ptr::null(),
        page_indptr: core::ptr::null(),
        last_page_lens: core::ptr::null(),
        key_scales: core::ptr::null(),
        value_scales: core::ptr::null(),
        write_page: core::ptr::null(),
        write_offset: core::ptr::null(),
        page_size: 16,
        seq_stride: 0,
        head_stride: 0,
        layout: 0,
        storage_dtype: 0,
        scheme_byte: 0,
        native_bf16: true,
        has_envelopes: false,
        env_min: core::ptr::null(),
        env_max: core::ptr::null(),
        block_size: 0,
        max_pages_per_request: 0,
        pages_in_batch: 0,
        qo_indptr: core::ptr::null(),
        row_valid: core::ptr::null(),
        requests: 0,
    }
}

/// What the routine derives off its own column, checked by what it refuses.
///
/// The rope width is `q_pe.width / heads` and the head count is stated;
/// nothing in the operands spells the rope width on its own, so a query
/// whose rotated half does not divide by the stated heads is a statement
/// this plane cannot serve — and answering it with a silently truncated
/// pitch would attend over the wrong bytes. The plan raise is a null the
/// executor forgot to stage, and it is named rather than dereferenced.
#[test]
fn the_column_refuses_what_it_cannot_derive() {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let view = blank_view();
    let plan = MlaPlan {
        info: Default::default(),
        int_arena: core::ptr::null_mut(),
        float_arena: core::ptr::null_mut(),
    };
    let q = |width: i32| In::<kernels_cuda::jit::abi::Tensor<bf16>> {
        ptr: core::ptr::null(),
        rows: 1,
        width,
    };
    let o = Out::<kernels_cuda::jit::abi::Tensor<bf16>> {
        ptr: core::ptr::null_mut(),
        rows: 1,
        width: 16 * CKV,
    };
    let row = |v: &PagedKvView| In::<Struct<KvCache>> {
        ptr: core::ptr::from_ref(v),
        rows: 0,
        width: 0,
    };
    let raised = |p: *const MlaPlan| In::<Struct<kernels_cuda::raises::MlaPlanned>> {
        ptr: p,
        rows: 0,
        width: 0,
    };

    // A rotated half that does not divide by the stated head count.
    let ragged = kernels_cuda::attn::attention_mla_decode_bf16(
        &ctx,
        q(16 * CKV),
        raised(core::ptr::from_ref(&plan)),
        q(16 * KPE - 1),
        o,
        row(&view),
        Const::new(16),
        Const::new(CKV),
        Const::new(0.5),
    );
    assert!(
        matches!(ragged, Err(kernels::Refusal::Narrow { .. })),
        "a ragged rotated half must be refused, not divided: {ragged:?}"
    );

    // A plan the executor did not stage.
    let unplanned = kernels_cuda::attn::attention_mla_decode_bf16(
        &ctx,
        q(16 * CKV),
        raised(core::ptr::null()),
        q(16 * KPE),
        o,
        row(&view),
        Const::new(16),
        Const::new(CKV),
        Const::new(0.5),
    );
    assert!(
        matches!(unplanned, Err(kernels::Refusal::Null { .. })),
        "an unstaged plan must be named, not dereferenced: {unplanned:?}"
    );

    // A pool row the executor did not bind.
    let unbound = kernels_cuda::attn::attention_mla_decode_bf16(
        &ctx,
        q(16 * CKV),
        raised(core::ptr::from_ref(&plan)),
        q(16 * KPE),
        o,
        In::<Struct<KvCache>> {
            ptr: core::ptr::null(),
            rows: 0,
            width: 0,
        },
        Const::new(16),
        Const::new(CKV),
        Const::new(0.5),
    );
    assert!(
        matches!(unbound, Err(kernels::Refusal::Null { .. })),
        "an unbound pool row must be named: {unbound:?}"
    );

    // And the append's own: a pool row carrying no CSR cannot resolve a
    // destination, which is the field `W7` put on the view.
    let no_csr = Mla::kv_append::<bf16>(
        &ctx,
        q(8),
        q(4),
        Cache::<Struct<KvCache>> {
            ptr: core::ptr::from_ref(&view),
        },
    );
    assert!(
        matches!(no_csr, Err(kernels::Refusal::Null { .. })),
        "an append with no query CSR on its pool row must be named: {no_csr:?}"
    );
}

// ── the attention over what the append left ──────────────────────────────

/// The FA2 MLA kernel's compiled-in latent and rope widths
/// (`attention_mla_fa2.cuh:168-169`). A test at any other pair would be
/// testing a kernel that does not exist.
const CKV: i32 = 512;
const KPE: i32 = 64;

/// The fa2 planner's two carves, as `driver-cuda/src/fire/launch.rs`
/// takes them for a whole deployment.
const FLOAT_BYTES: usize = 32 << 20;
const INT_BYTES: usize = 16 << 20;

/// Whether the arm this box would pick is one that runs.
///
/// THE NARROWEST FA2 ARM IS BROKEN AND THE PLANE NOW REFUSES IT, so on a
/// device that can only pick that arm there is no arithmetic to check and
/// this says so rather than asserting a refusal is a result. The measurement
/// that put the refusal there is written down at the refusal
/// (`dispatch_attention_mla_bf16`); it was made by this very test on an
/// L40S, under `compute-sanitizer`.
///
/// sm_100 and up never reach the FA2 arms at all — they take the naive
/// kernel, which has no shared-storage question — so the gate is only asked
/// below that.
fn this_device_can_attend_in_the_latent_basis() -> bool {
    use kernels_cuda::attn::{fa2, mla_fa2};
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    if ctx.compute_capability_major().is_none_or(|m| m >= 10) {
        return true;
    }
    let smem = fa2::plan::fa_device().max_smem_per_sm;
    match mla_fa2::arm_for(smem) {
        None => {
            eprintln!(
                "skipping mla.attention_decode: no `DISPATCH_SMEM_CONFIG` arm fits this device's {smem} B of shared memory per SM"
            );
            false
        }
        Some(arm) if arm.cta_tile_kv < 32 => {
            eprintln!(
                "skipping mla.attention_decode: this device's {smem} B of shared memory per SM picks the `CTA_TILE_KV = 16` arm, which writes past its own `SharedStorage` and which the plane refuses by name"
            );
            false
        }
        Some(_) => true,
    }
}

/// One decode fire: `heads` query rows against one request's cached keys.
struct Decode {
    heads: i32,
    page_size: i32,
    kv_len: i32,
}

impl Decode {
    fn pages(&self) -> i32 {
        (self.kv_len + self.page_size - 1) / self.page_size
    }
}

#[test]
fn the_decode_attends_the_pages_it_was_given() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("mla.attention_decode") {
        return;
    }
    if !this_device_can_attend_in_the_latent_basis() {
        return;
    }
    let d = Decode {
        heads: 16,
        page_size: 16,
        kv_len: 21,
    };
    let mut rng = Rng(0x1357_9bdf_0246_8ace);
    let pages = d.pages();
    // The pool holds more pages than the request owns, and the request's
    // pages are scattered through it.
    let pool_pages = pages + 3;
    let page_indices: Vec<u32> = (0..pages as u32).map(|p| pool_pages as u32 - 1 - p).collect();

    let ckv_elems = (pool_pages * d.page_size * CKV) as usize;
    let kpe_elems = (pool_pages * d.page_size * KPE) as usize;
    let mut h_ckv = vec![0u16; ckv_elems];
    let mut h_kpe = vec![0u16; kpe_elems];
    for j in 0..d.kv_len {
        let page = page_indices[(j / d.page_size) as usize] as i32;
        let off = j % d.page_size;
        let c0 = ((page * d.page_size + off) * CKV) as usize;
        let p0 = ((page * d.page_size + off) * KPE) as usize;
        for i in 0..CKV as usize {
            h_ckv[c0 + i] = rng.bf16(1.0);
        }
        for i in 0..KPE as usize {
            h_kpe[p0 + i] = rng.bf16(1.0);
        }
    }

    let q_nope: Vec<u16> = (0..(d.heads * CKV) as usize).map(|_| rng.bf16(0.5)).collect();
    let q_pe: Vec<u16> = (0..(d.heads * KPE) as usize).map(|_| rng.bf16(0.5)).collect();
    let sm_scale = 1.0f32 / ((CKV + KPE) as f32).sqrt();

    // The host reference: one softmax per head over the request's keys, with
    // the latent row standing as both key and value.
    let mut want = vec![0.0f32; (d.heads * CKV) as usize];
    for h in 0..d.heads as usize {
        let qn = &q_nope[h * CKV as usize..(h + 1) * CKV as usize];
        let qp = &q_pe[h * KPE as usize..(h + 1) * KPE as usize];
        let mut scores = vec![0.0f32; d.kv_len as usize];
        for j in 0..d.kv_len {
            let page = page_indices[(j / d.page_size) as usize] as i32;
            let off = j % d.page_size;
            let c0 = ((page * d.page_size + off) * CKV) as usize;
            let p0 = ((page * d.page_size + off) * KPE) as usize;
            let mut s = 0.0f32;
            for i in 0..CKV as usize {
                s += wide(qn[i]) * wide(h_ckv[c0 + i]);
            }
            for i in 0..KPE as usize {
                s += wide(qp[i]) * wide(h_kpe[p0 + i]);
            }
            scores[j as usize] = s * sm_scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for j in 0..d.kv_len as usize {
            scores[j] = (scores[j] - m).exp();
            denom += scores[j];
        }
        for j in 0..d.kv_len {
            let page = page_indices[(j / d.page_size) as usize] as i32;
            let off = j % d.page_size;
            let c0 = ((page * d.page_size + off) * CKV) as usize;
            let p = scores[j as usize] / denom;
            for i in 0..CKV as usize {
                want[h * CKV as usize + i] += p * wide(h_ckv[c0 + i]);
            }
        }
    }

    // The pool row. ONE REQUEST, so the CSR is `[0, 1]` and the whole cached
    // prefix is this fire's context.
    let d_ckv = Slab::of(&bytes_of_u16(&h_ckv));
    let d_kpe = Slab::of(&bytes_of_u16(&h_kpe));
    let d_indices = Slab::of(&bytes_of_u32(&page_indices));
    let d_indptr = Slab::of(&bytes_of_u32(&[0, pages as u32]));
    let last = d.kv_len - (pages - 1) * d.page_size;
    let d_lens = Slab::of(&bytes_of_u32(&[last as u32]));
    let d_qo = Slab::of(&bytes_of_u32(&[0, 1]));
    let view = PagedKvView {
        keys: d_ckv.ptr.cast(),
        values: d_kpe.ptr.cast(),
        bf16_keys: d_ckv.ptr.cast(),
        bf16_values: d_kpe.ptr.cast(),
        page_indices: d_indices.ptr.cast(),
        page_indptr: d_indptr.ptr.cast(),
        last_page_lens: d_lens.ptr.cast(),
        key_scales: core::ptr::null(),
        value_scales: core::ptr::null(),
        write_page: core::ptr::null(),
        write_offset: core::ptr::null(),
        page_size: d.page_size,
        seq_stride: 0,
        head_stride: 0,
        layout: 0,
        storage_dtype: kernels_cuda::attn::KvDType::Bf16 as i32,
        scheme_byte: kernels_cuda::attn::KvScheme::Native as i32,
        native_bf16: true,
        has_envelopes: false,
        env_min: core::ptr::null(),
        env_max: core::ptr::null(),
        block_size: 0,
        max_pages_per_request: pages,
        pages_in_batch: pages,
        qo_indptr: d_qo.ptr.cast(),
        row_valid: core::ptr::null(),
        requests: 1,
    };

    // THE PLANE'S STAGING, done here because that is the whole reason this
    // point is claim-only: the schedule is measured on the HOST out of the
    // three CSRs and uploaded into an int arena the launch reads.
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let device = Device::new(
        ctx.multiprocessors().expect("the device's SM count"),
        ctx.compute_capability_major()
            .expect("the device's compute capability")
            .cast_signed(),
    );
    let qo_h = [0i32, 1];
    let kv_h = [0i32, pages];
    let lens_h = [d.kv_len];
    let planned = plan::mla::plan(
        &plan::mla::Request {
            qo_indptr: &qo_h,
            kv_indptr: &kv_h,
            kv_len_arr: &lens_h,
            batch_size: 1,
            num_heads: d.heads.unsigned_abs(),
            head_dim_o: CKV.unsigned_abs(),
            causal: false,
        },
        &device,
        // The driver's own carve (`fire/launch.rs:1494-1510`, copied into
        // `baker-smoke`): 32 MiB float and 16 MiB int for a whole
        // deployment. `Workspace::unbounded()` cannot be used here — the
        // planner STAGES the int arena into a host `Vec` of exactly that
        // length before it truncates to what it used.
        Workspace::new(FLOAT_BYTES, INT_BYTES),
    )
    .expect("a latent attention schedule for this shape");
    // THE ARENA IS THE WORKSPACE, NOT THE UPLOAD. `mla_fa2::pack` turns each
    // `*_offset` into `int_buffer.cast::<i32>().offset(o)` — an ELEMENT
    // offset applied to a BYTE offset, so every address it hands the kernel
    // is four times as far in as the plan measured. The driver never sees it
    // because it carves 16 MiB once for a whole deployment and the plan uses
    // a fraction of it; a test that allocated exactly `int_upload.len()`
    // walks off the end on the first read. Same carve here, same reason, and
    // the seam is named rather than papered over.
    let d_int = Slab::zeroed(INT_BYTES);
    d_int.upload(&planned.int_upload);
    let d_float = Slab::zeroed(FLOAT_BYTES);
    eprintln!(
        "mla plan: {} int byte(s) uploaded, {} float byte(s) carved, grid {}x{}",
        planned.int_upload.len(),
        planned.float_bytes,
        planned.info.num_blks_x,
        planned.info.num_blks_y
    );
    let mla_plan = MlaPlan {
        info: planned.info,
        int_arena: d_int.ptr,
        float_arena: d_float.ptr,
    };

    let d_q = Slab::of(&bytes_of_u16(&q_nope));
    let d_qpe = Slab::of(&bytes_of_u16(&q_pe));
    let d_out = Slab::zeroed((d.heads * CKV) as usize * 2);

    kernels_cuda::attn::attention_mla_decode_bf16(
        &ctx,
        In {
            ptr: d_q.ptr.cast(),
            rows: 1,
            width: d.heads * CKV,
        },
        In {
            ptr: core::ptr::from_ref(&mla_plan),
            rows: 0,
            width: 0,
        },
        In {
            ptr: d_qpe.ptr.cast(),
            rows: 1,
            width: d.heads * KPE,
        },
        Out {
            ptr: d_out.ptr.cast(),
            rows: 1,
            width: d.heads * CKV,
        },
        In {
            ptr: core::ptr::from_ref(&view),
            rows: 0,
            width: 0,
        },
        Const::new(d.heads),
        Const::new(CKV),
        Const::new(sm_scale),
    )
    .expect("`mla.attention_decode`'s columned routine");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the attention did not complete"
    );

    let got = d_out.read_u16((d.heads * CKV) as usize);
    let (mut num, mut den, mut worst) = (0.0f64, 0.0f64, 0.0f32);
    for (g, w) in got.iter().zip(want.iter()) {
        let e = wide(*g) - *w;
        num += f64::from(e) * f64::from(e);
        den += f64::from(*w) * f64::from(*w);
        worst = worst.max(e.abs());
    }
    let rms = (num / den.max(1e-30)).sqrt();
    eprintln!("mla.attention_decode: relative rms {rms:.3e}, worst |err| {worst:.3e}");
    // THE BAR IS BF16'S. The reference sums 512 products in f32 in index
    // order; the kernel sums them in a different order and stores the result
    // as bf16, whose own resolution is about `4e-3` relative. `2e-2` is what
    // that costs and is two orders away from a wrong page.
    assert!(
        rms < 2e-2,
        "relative rms {rms:.3e} against the host softmax over the same pages"
    );
}
