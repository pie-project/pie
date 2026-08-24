//! `attention.kv_append_shared`'s claimed body: dsv4's ONE plane, landing in
//! both halves of a pool row.
//!
//! THE ALIAS IS WHAT IS UNDER TEST. The body hands `write_kv_to_pages_bf16`
//! one address where the kernel declares two `const bf16* __restrict__`
//! sources, so the check that matters is that BOTH page planes come back
//! holding the same rows the statement was handed — a key plane written and
//! a value plane left poisoned is what an alias the kernel could not take
//! would look like, and it is not something a shape test would notice.
//!
//! The rest is `mla_paged.rs`'s reading of the same seam, and for its
//! reason: the point is claimed by a BODY, so there is no routine whose row
//! a table test could check, and what can be checked is where the bytes
//! land. The reference is the destination walk written out on the host —
//! `pre_kv_len = total_kv_after - new_tokens` (`attn/kv_paged.cuh:172-187`)
//! — and a body that read the pool row's planes in the wrong order, took the
//! head split off the wrong place, or dropped the fire's row validity fails
//! it and passes nothing else.
//!
//! # The head split has no operand
//!
//! `kv_append_shared(plane, pages)` states no head count and no head width,
//! because a shared plane has neither: dsv4's `kv_down` is
//! `[heads * head_dim, hidden]` and what comes out is one rectangle. The
//! body reads the split off `PagedKvView::seq_stride`/`head_stride`, which
//! is where the POOL recorded it. The toy below therefore lays its pool out
//! at a head split the plane's width alone cannot spell — 3 heads of 8, not
//! 1 of 24 — and the last test asserts the refusals a row whose strides do
//! not divide gets.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Attention;
use kernels::raises::Struct;
use kernels::routine::{Cache, In};
use kernels::Refusal;
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

/// One dense pool and the fire that writes into it.
///
/// EVERY NUMBER HERE IS AWKWARD ON PURPOSE, `mla_paged.rs`'s toy at a
/// different geometry: the page indices are scattered rather than `0..n`,
/// the two requests own different page counts, the last page of each is
/// partly full, and one row is INVALID.
struct Toy {
    kv_heads: i32,
    head_dim: i32,
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
            // 3 x 8 = 24, and the plane's width alone would read as 1 x 24.
            // The pool's strides are the only place the split lives.
            kv_heads: 3,
            head_dim: 8,
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

    fn row(&self) -> i32 {
        self.kv_heads * self.head_dim
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
    /// (`attn/kv_paged.cuh:172-187`) and of nothing else.
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
        // A POISON FILL, not zeros: a body that wrote nothing at all would
        // pass a zero-vs-zero comparison at every slot it was supposed to
        // skip, and the reference below expects the poison to survive.
        let keys = Slab::of(&bytes_of_u16(&vec![narrow(-7.5); toy.plane_elems()]));
        let values = Slab::of(&bytes_of_u16(&vec![narrow(-3.25); toy.plane_elems()]));
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
            // NHD, in ELEMENTS, per `driver-cuda/src/bind/views.rs:336-346`:
            // a page is `[page_size, kv_heads, head_dim]`, so a token step
            // crosses every head and a head is `head_dim`. THIS PAIR IS THE
            // BODY'S ONLY SOURCE for the head split.
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

/// Fire `attention.kv_append_shared` over `toy`.
fn append(toy: &Toy, pool: &Pool, plane: &[u16]) {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let d_plane = Slab::of(&bytes_of_u16(plane));
    Attention::kv_append_shared::<bf16>(
        &ctx,
        In {
            ptr: d_plane.ptr.cast(),
            rows: toy.rows(),
            width: toy.row(),
        },
        pool.cache(),
    )
    .expect("the claimed `attention.kv_append_shared` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the append did not complete"
    );
}

#[test]
fn one_plane_lands_in_both_halves_of_the_row() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("attention.kv_append_shared") {
        return;
    }
    let toy = Toy::small();
    let pool = Pool::build(&toy);
    let mut rng = Rng(0x5eed_4321_1234_9876);
    let plane: Vec<u16> = (0..(toy.rows() * toy.row()) as usize)
        .map(|_| rng.bf16(2.0))
        .collect();

    append(&toy, &pool, &plane);

    // The reference: each plane's poison, overwritten only where a VALID
    // row's destination says — and the SAME source row in both, which is
    // the whole of what "shared" means.
    let mut want_k = vec![narrow(-7.5); toy.plane_elems()];
    let mut want_v = vec![narrow(-3.25); toy.plane_elems()];
    for t in 0..toy.rows() {
        if toy.row_valid[t as usize] == 0 {
            continue;
        }
        let (page, off) = toy.destination(t);
        let dst = ((page * toy.page_size + off) * toy.row()) as usize;
        for i in 0..toy.row() as usize {
            let src = plane[t as usize * toy.row() as usize + i];
            want_k[dst + i] = src;
            want_v[dst + i] = src;
        }
    }

    let elems = toy.plane_elems();
    let got_k = pool.keys.read_u16(elems);
    let got_v = pool.values.read_u16(elems);
    let bad_k = (0..elems).filter(|i| got_k[*i] != want_k[*i]).count();
    let bad_v = (0..elems).filter(|i| got_v[*i] != want_v[*i]).count();
    eprintln!(
        "attention.kv_append_shared: keys {}/{elems} exact, values {}/{elems} exact",
        elems - bad_k,
        elems - bad_v
    );
    // A COPY IS EXACT OR IT IS WRONG. There is no arithmetic in this kernel
    // to give a tolerance to.
    assert_eq!(bad_k, 0, "{bad_k} element(s) landed wrong in the key plane");
    assert_eq!(
        bad_v, 0,
        "{bad_v} element(s) landed wrong in the value plane — the alias did \
         not reach the second destination"
    );
}

/// The two rows the body must refuse before it touches the device.
///
/// Both are read on the HOST — a `Cache` mark carries a struct pointer and
/// nothing else — so a view with dangling device pointers is exactly what
/// these paths see, and they are checkable with no fire at all.
#[test]
fn the_body_refuses_what_the_pool_row_does_not_spell() {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let toy = Toy::small();
    let blank = |head_stride: i64, qo: *const i32| PagedKvView {
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
        page_size: toy.page_size,
        seq_stride: i64::from(toy.kv_heads) * i64::from(toy.head_dim),
        head_stride,
        layout: 0,
        storage_dtype: kernels_cuda::attn::KvDType::Bf16 as i32,
        scheme_byte: kernels_cuda::attn::KvScheme::Native as i32,
        native_bf16: true,
        has_envelopes: false,
        env_min: core::ptr::null(),
        env_max: core::ptr::null(),
        block_size: 0,
        max_pages_per_request: 0,
        pages_in_batch: 0,
        qo_indptr: qo,
        row_valid: core::ptr::null(),
        requests: toy.requests(),
    };
    let plane = |width: i32| In::<kernels_cuda::jit::abi::Tensor<bf16>> {
        ptr: core::ptr::null(),
        rows: toy.rows(),
        width,
    };
    let cache = |v: &PagedKvView| Cache::<Struct<KvCache>> {
        ptr: core::ptr::from_ref(v),
    };

    // A pool row with no query CSR is a fire the driver did not stage; the
    // body names it rather than dereferencing it.
    let no_csr = blank(i64::from(toy.head_dim), core::ptr::null());
    assert_eq!(
        Attention::kv_append_shared::<bf16>(&ctx, plane(toy.row()), cache(&no_csr)),
        Err(Refusal::Null {
            what: "the query CSR this fire's pool row carries",
        })
    );

    // A plane that does not divide by the pool's head width is a statement
    // this plane cannot serve: the write would stripe the wrong heads and
    // every launch would still succeed.
    let ok = blank(i64::from(toy.head_dim), core::ptr::from_ref(&toy).cast());
    assert_eq!(
        Attention::kv_append_shared::<bf16>(&ctx, plane(toy.row() - 1), cache(&ok)),
        Err(Refusal::Narrow {
            what: "the appended plane does not divide by the pool row's head width",
            at: i64::from(toy.row() - 1),
        })
    );

    // A pool row whose strides are unset says nothing about the split, and
    // a body that guessed `1 x width` would agree with the NHD arithmetic
    // by accident and disagree with an HND pool silently.
    let unset = blank(0, core::ptr::from_ref(&toy).cast());
    assert_eq!(
        Attention::kv_append_shared::<bf16>(&ctx, plane(toy.row()), cache(&unset)),
        Err(Refusal::Empty {
            what: "the head width this pool row's strides spell",
        })
    );

    // And the element pin, stated by name rather than widened with a cast
    // no kernel stands behind.
    assert_eq!(
        Attention::kv_append_shared::<f32>(
            &ctx,
            In::<kernels_cuda::jit::abi::Tensor<f32>> {
                ptr: core::ptr::null(),
                rows: toy.rows(),
                width: toy.row(),
            },
            cache(&ok),
        ),
        Err(Refusal::Absent {
            what: "attention.kv_append_shared at an element other than bf16",
        })
    );
}
