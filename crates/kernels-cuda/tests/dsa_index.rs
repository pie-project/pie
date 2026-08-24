//! glm's DSA selection path, claimed: `index.kv_append`, `index.topk`, and
//! the two `mla.attention_*_selected` readings that consume what they make.
//!
//! THERE IS NO SHIPPED GROUND TRUTH FOR ANY OF IT, and that is the reason
//! this file is written the way it is. The legacy glm text projected the
//! index keys, rotated them, scored the TOKEN PLANE it had just written, and
//! assigned the mask to `let _index_mask` — the selection was computed and
//! thrown away, and nothing downstream ever read one. So there is no
//! A/B partner to diff against, and the reference for each point below is
//! the arithmetic written out on the host:
//!
//!  * `index.kv_append` — the destination walk, `pre_kv_len =
//!    total_kv_after - new_tokens` (`attn/mla_paged.cuh`'s
//!    `mla_resolve_dst`), which is `mla_paged.rs`'s reference at one plane
//!    instead of two. A COPY IS EXACT OR IT IS WRONG.
//!  * `index.topk` — the DSA ranking as the papers spell it and as
//!    `index_topk_mask` already computed it,
//!    `logit[t, j] = Σ_h relu(q[t, h] · k[j]) * w[t, h]`, causal in the
//!    request's ABSOLUTE positions and over the PAGED keys. The reference
//!    ranks by value and the kernel thresholds by bisection, so the two
//!    agree on the SET, which is the whole content of a selection.
//!  * the two selected attentions — one softmax per head over the keys the
//!    selection keeps, at `mla_paged.rs`'s bf16 bar.
//!
//! # Every reference has a mutation beside it
//!
//! A reference that merely re-derives what the kernel does can agree with it
//! for the wrong reason, so each check below is paired with the WRONG
//! reading it must reject:
//!
//!  * the append: one that ignores `row_valid`, and one that starts each
//!    request at page offset zero;
//!  * the ranking: one that is not causal (every cached key ranked, not just
//!    `j <= abs_q`), and the LEGACY one — the token-plane reading, where `j`
//!    indexes this fire's own rows instead of the pool's pages. That second
//!    mutation is the survey's finding turned into an assertion: if the
//!    paged kernel agreed with the token-plane reading, the pool would not be
//!    being read at all.
//!  * the selected attention: one that attends every key in the causal
//!    range rather than the selected ones.
//!
//! # This runs on this box
//!
//! `mla.attention_{decode,prefill}_selected` do NOT branch on compute
//! capability, because only one latent attention kernel in the tree can walk
//! a selection: `mla_naive_paged_kernel`. The FA2 MLA kernel has no
//! selection in `MlaParams` at all, and the tensor-core arm stages `kBK`
//! CONTIGUOUS keys through one `cp.async` copy. So the L40S-class refusal
//! `mla_paged.rs` has to skip around — the `CTA_TILE_KV = 16` arm that
//! writes past its own `SharedStorage` — is not on this path, and these two
//! points are numerically checked on hardware where the unselected pair is
//! not.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::{Index, Mla};
use kernels::raises::Struct;
use kernels::routine::{Cache, In, Out};
use kernels::Refusal;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::jit::Ctx;
use kernels_cuda::views::{KvCache, PagedKvView};

/// The device scratch is a process-global named-slab arena sized for one
/// fire at a time, which the driver's stream serialization guarantees and a
/// test harness's thread pool does not. `index.topk`'s logits ride one, so
/// this lock is load-bearing here and not a copied habit.
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

    fn read_i32(&self, elems: usize) -> Vec<i32> {
        self.read(elems * 4)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

// ── the toy fire ─────────────────────────────────────────────────────────

/// One paged pool and the fire that writes into it.
///
/// EVERY NUMBER IS AWKWARD ON PURPOSE, `mla_paged.rs`'s toy at the indexer's
/// geometry: the page indices are scattered rather than `0..n`, the two
/// requests own different page counts, each request's last page is partly
/// full, the two requests carry different numbers of new rows, and one row
/// is INVALID. A body that ignored `row_valid`, walked the pages in request
/// order, or assumed the append starts at offset zero passes none of it.
#[derive(Clone)]
struct Toy {
    /// The indexer's key width — one head, `[1, head_dim]`, which is what
    /// glm's `KvSpec` declares for `index.{l}`.
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
    /// A prefill-shaped fire: request 0 adds 2 rows to a 4-row prefix,
    /// request 1 adds 3 rows to an empty one.
    fn prefill() -> Toy {
        Toy {
            head_dim: 8,
            page_size: 4,
            pool_pages: 8,
            qo_indptr: vec![0, 2, 5],
            page_indptr: vec![0, 2, 3],
            page_indices: vec![5, 2, 7],
            last_page_lens: vec![2, 3],
            row_valid: vec![1, 1, 0, 1, 1],
        }
    }

    /// A decode-shaped fire: one new row per request, onto prefixes of 6 and
    /// 3. Every row is valid — a decode that rejected one would carry the
    /// same null the driver stages.
    fn decode() -> Toy {
        Toy {
            head_dim: 8,
            page_size: 4,
            pool_pages: 8,
            qo_indptr: vec![0, 1, 2],
            page_indptr: vec![0, 2, 3],
            page_indices: vec![5, 2, 7],
            last_page_lens: vec![3, 4],
            row_valid: vec![1, 1],
        }
    }

    fn rows(&self) -> i32 {
        *self.qo_indptr.last().expect("a CSR ends somewhere") as i32
    }

    fn requests(&self) -> i32 {
        self.qo_indptr.len() as i32 - 1
    }

    fn key_elems(&self) -> usize {
        (self.pool_pages * self.page_size * self.head_dim) as usize
    }

    /// Which request row `t` belongs to.
    fn request_of(&self, t: i32) -> usize {
        (0..self.requests())
            .find(|r| t < self.qo_indptr[*r as usize + 1] as i32)
            .unwrap_or(self.requests() - 1) as usize
    }

    /// How many keys request `r` holds after this fire.
    fn kv_len(&self, r: usize) -> i32 {
        let pages = self.page_indptr[r + 1] as i32 - self.page_indptr[r] as i32;
        (pages - 1) * self.page_size + self.last_page_lens[r] as i32
    }

    /// The absolute cached position row `t` occupies.
    ///
    /// A transcription of `pie::attn::mla_resolve_dst`
    /// (`attn/mla_paged.cuh`) and of nothing else.
    fn absolute(&self, t: i32) -> (usize, i32) {
        let r = self.request_of(t);
        let qo_lo = self.qo_indptr[r] as i32;
        let new_tokens = self.qo_indptr[r + 1] as i32 - qo_lo;
        (r, self.kv_len(r) - new_tokens + (t - qo_lo))
    }

    /// Where key `j` of request `r` lives in the pool.
    fn slot(&self, r: usize, j: i32) -> usize {
        let first = self.page_indptr[r] as i32;
        let page = self.page_indices[(first + j / self.page_size) as usize] as i32;
        (page * self.page_size + j % self.page_size) as usize
    }
}

/// The pool's device planes and the view a statement names.
struct Pool {
    keys: Slab,
    _indices: Slab,
    _indptr: Slab,
    _lens: Slab,
    _qo: Slab,
    _valid: Slab,
    view: PagedKvView,
}

impl Pool {
    /// `fill` is the pool's key plane at construction: a poison for the
    /// append test, real keys for the ranking one.
    fn build(toy: &Toy, fill: &[u16], row_valid: bool) -> Pool {
        let keys = Slab::of(&bytes_of_u16(fill));
        let indices = Slab::of(&bytes_of_u32(&toy.page_indices));
        let indptr = Slab::of(&bytes_of_u32(&toy.page_indptr));
        let lens = Slab::of(&bytes_of_u32(&toy.last_page_lens));
        let qo = Slab::of(&bytes_of_u32(&toy.qo_indptr));
        let valid = Slab::of(&toy.row_valid);
        let view = PagedKvView {
            keys: keys.ptr.cast(),
            // THE INDEXER CACHES A KEY AND NO VALUE. The value plane is not
            // this pool's, and `index.kv_append`'s body hands the append a
            // NULL second page plane and a zero second width rather than
            // aliasing this one into it.
            values: core::ptr::null_mut(),
            bf16_keys: keys.ptr.cast(),
            bf16_values: core::ptr::null_mut(),
            page_indices: indices.ptr.cast(),
            page_indptr: indptr.ptr.cast(),
            last_page_lens: lens.ptr.cast(),
            key_scales: core::ptr::null(),
            value_scales: core::ptr::null(),
            write_page: core::ptr::null(),
            write_offset: core::ptr::null(),
            page_size: toy.page_size,
            // ONE HEAD OF `head_dim`, laid out NHD — which is what
            // `driver-cuda/src/bind/views.rs::kv_view` computes for a pool
            // declared `[1, head_dim]`, and what `index.kv_append` checks
            // its row against.
            seq_stride: i64::from(toy.head_dim),
            head_stride: i64::from(toy.head_dim),
            layout: 0,
            storage_dtype: kernels_cuda::attn::KvDType::Bf16 as i32,
            scheme_byte: kernels_cuda::attn::KvScheme::Native as i32,
            native_bf16: true,
            has_envelopes: false,
            env_min: core::ptr::null(),
            env_max: core::ptr::null(),
            block_size: 0,
            max_pages_per_request: toy.pool_pages,
            pages_in_batch: toy.page_indices.len() as i32,
            qo_indptr: qo.ptr.cast(),
            row_valid: if row_valid {
                valid.ptr.cast()
            } else {
                core::ptr::null()
            },
            requests: toy.requests(),
        };
        Pool {
            keys,
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

// ── index.kv_append ──────────────────────────────────────────────────────

#[test]
fn the_index_append_lands_where_the_csr_says() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("index.kv_append") {
        return;
    }
    let toy = Toy::prefill();
    // A POISON FILL, not zeros: a body that wrote nothing at all would pass
    // a zero-vs-zero comparison at every slot it was supposed to skip.
    let poison = narrow(-7.5);
    let pool = Pool::build(&toy, &vec![poison; toy.key_elems()], true);

    let mut rng = Rng(0x0d5a_1234_9876_4321);
    let rows = toy.rows() as usize;
    let k: Vec<u16> = (0..rows * toy.head_dim as usize)
        .map(|_| rng.bf16(2.0))
        .collect();
    let d_k = Slab::of(&bytes_of_u16(&k));

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    Index::kv_append::<bf16>(
        &ctx,
        In {
            ptr: d_k.ptr.cast(),
            rows: toy.rows(),
            width: toy.head_dim,
        },
        pool.row(),
    )
    .expect("the claimed `index.kv_append` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the append did not complete"
    );

    let got = pool.keys.read_u16(toy.key_elems());

    // The reference: the poison, overwritten only where a VALID row's
    // destination says.
    let want = appended(&toy, &k, poison, true, true);
    let bad = (0..toy.key_elems()).filter(|i| got[*i] != want[*i]).count();
    eprintln!(
        "index.kv_append: {}/{} element(s) exact",
        toy.key_elems() - bad,
        toy.key_elems()
    );
    // A COPY IS EXACT OR IT IS WRONG. There is no arithmetic in this kernel
    // to give a tolerance to.
    assert_eq!(bad, 0, "{bad} index key element(s) landed wrong");

    // THE MUTATIONS. Each is a reading the body could plausibly have had,
    // and each must DISAGREE with what the device wrote — a reference that
    // agreed with every wrong reading would be measuring nothing.
    let deaf = appended(&toy, &k, poison, false, true);
    assert_ne!(
        got, deaf,
        "a body that ignored the fire's row validity would land the same bytes"
    );
    let at_zero = appended(&toy, &k, poison, true, false);
    assert_ne!(
        got, at_zero,
        "a body that started each request at page offset zero would land the same bytes"
    );
}

/// The pool's key plane after an append, on the host.
///
/// `honour_valid` and `honour_prefix` are the two mutation switches: with
/// both true this is the kernel's own destination walk.
fn appended(
    toy: &Toy,
    k: &[u16],
    poison: u16,
    honour_valid: bool,
    honour_prefix: bool,
) -> Vec<u16> {
    let mut want = vec![poison; toy.key_elems()];
    for t in 0..toy.rows() {
        if honour_valid && toy.row_valid[t as usize] == 0 {
            continue;
        }
        let (r, abs) = toy.absolute(t);
        let at = if honour_prefix {
            abs
        } else {
            t - toy.qo_indptr[r] as i32
        };
        let dst = toy.slot(r, at) * toy.head_dim as usize;
        for i in 0..toy.head_dim as usize {
            want[dst + i] = k[t as usize * toy.head_dim as usize + i];
        }
    }
    want
}

// ── index.topk ───────────────────────────────────────────────────────────

/// The indexer's query geometry for the ranking tests.
const HEADS: i32 = 3;
const TOP_K: i32 = 3;

/// `logit[t, j]` for every cached key `j` of `t`'s request, on the host.
///
/// `causal` and `paged` are the two mutation switches; with both true this
/// is `pie::attn::index_topk_paged`'s own scoring.
fn logits(
    toy: &Toy,
    keys: &[u16],
    q: &[u16],
    w: &[u16],
    t: i32,
    causal: bool,
    paged: bool,
) -> Vec<f32> {
    let (r, abs) = toy.absolute(t);
    let nkeys = if causal { abs + 1 } else { toy.kv_len(r) };
    let d = toy.head_dim as usize;
    (0..nkeys)
        .map(|j| {
            // THE MUTATION IS THE ADDRESS, not the arithmetic: the legacy
            // reading scored the fire's own token plane, where `j` is a row
            // of this fire; the paged one resolves `j` through the pool.
            let base = if paged {
                toy.slot(r, j) * d
            } else {
                (toy.qo_indptr[r] as usize + j as usize) * d
            };
            let mut acc = 0.0f32;
            for h in 0..HEADS as usize {
                let qh = &q[(t as usize * HEADS as usize + h) * d..][..d];
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += wide(qh[i]) * wide(keys[base + i]);
                }
                acc += dot.max(0.0) * wide(w[t as usize * HEADS as usize + h]);
            }
            acc
        })
        .collect()
}

/// The `top_k` largest of `scores`, ascending, `-1` past the end.
///
/// RANKED BY VALUE, not by the kernel's bisection. The kernel finds a
/// threshold by forty halvings of the logit range and admits `>= thr`; this
/// takes the largest `top_k` outright. The two agree exactly when the k-th
/// and (k+1)-th logits are distinct floats, which the caller asserts — so
/// this is the SEMANTIC reference and not a transcription of the method.
fn chosen(scores: &[f32], top_k: i32) -> Vec<i32> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .expect("no NaN in a toy's logits")
            .then(a.cmp(b))
    });
    let mut keep: Vec<i32> = order
        .iter()
        .take(top_k as usize)
        .map(|j| *j as i32)
        .collect();
    keep.sort_unstable();
    keep.resize(top_k as usize, -1);
    keep
}

#[test]
fn the_ranking_reads_the_pool_and_answers_the_selection() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("index.topk") {
        return;
    }
    let toy = Toy::prefill();
    let mut rng = Rng(0x7071_c0de_4242_1001);
    // A pool FULL of keys: the ranking reads the cached prefix, not just the
    // rows this fire appended, and a poisoned prefix would let a kernel that
    // never left the batch pass.
    let keys: Vec<u16> = (0..toy.key_elems()).map(|_| rng.bf16(1.0)).collect();
    let pool = Pool::build(&toy, &keys, true);

    let rows = toy.rows() as usize;
    let d = toy.head_dim as usize;
    let q: Vec<u16> = (0..rows * HEADS as usize * d)
        .map(|_| rng.bf16(1.0))
        .collect();
    let w: Vec<u16> = (0..rows * HEADS as usize).map(|_| rng.bf16(1.0)).collect();

    let d_q = Slab::of(&bytes_of_u16(&q));
    let d_w = Slab::of(&bytes_of_u16(&w));
    let d_sel = Slab::zeroed(rows * TOP_K as usize * 4);

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    Index::topk::<bf16>(
        &ctx,
        In {
            ptr: d_q.ptr.cast(),
            rows: toy.rows(),
            width: HEADS * toy.head_dim,
        },
        In {
            ptr: d_w.ptr.cast(),
            rows: toy.rows(),
            width: HEADS,
        },
        pool.row(),
        HEADS.unsigned_abs(),
        toy.head_dim.unsigned_abs(),
        TOP_K.unsigned_abs(),
        Out {
            ptr: d_sel.ptr.cast(),
            rows: toy.rows(),
            width: TOP_K,
        },
    )
    .expect("the claimed `index.topk` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the ranking did not complete"
    );
    let got = d_sel.read_i32(rows * TOP_K as usize);

    let mut want = Vec::new();
    let mut not_causal = Vec::new();
    let mut token_plane = Vec::new();
    // How many rows actually RANK. A row whose causal prefix is no longer
    // than the budget takes the kernel's `nkeys <= topk` early-out and
    // proves nothing about the bisection, so the toy is asserted to have
    // some that do not.
    let mut ranking = 0;
    for t in 0..toy.rows() {
        let l = logits(&toy, &keys, &q, &w, t, true, true);
        if l.len() > TOP_K as usize {
            ranking += 1;
        }
        // THE BOUNDARY MUST BE A REAL ONE. The kernel's bisection converges
        // on the k-th value and admits `>= thr`; a row whose k-th and
        // (k+1)-th logits were the same float would let it admit either, and
        // the reference's ranking would be arbitrary too. The toy's are not.
        if l.len() > TOP_K as usize {
            let mut sorted = l.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).expect("no NaN"));
            assert_ne!(
                sorted[TOP_K as usize - 1],
                sorted[TOP_K as usize],
                "row {t}'s selection boundary is a tie; the toy is not separating"
            );
        }
        want.extend(chosen(&l, TOP_K));
        not_causal.extend(chosen(&logits(&toy, &keys, &q, &w, t, false, true), TOP_K));
        token_plane.extend(chosen(&logits(&toy, &keys, &q, &w, t, true, false), TOP_K));
    }

    assert!(
        ranking >= 2,
        "only {ranking} row(s) of the toy exceed the budget; the bisection is \
         barely exercised"
    );
    eprintln!("index.topk: {ranking} of {} row(s) rank", toy.rows());
    eprintln!("index.topk: got {got:?}");
    eprintln!("index.topk: want {want:?}");
    assert_eq!(
        got, want,
        "the selection is not the top {TOP_K} logits of each row's cached prefix"
    );

    // THE MUTATIONS.
    assert_ne!(
        got, not_causal,
        "a ranking over every cached key would answer the same selection"
    );
    assert_ne!(
        got, token_plane,
        "the LEGACY token-plane reading would answer the same selection — the \
         pool is not being read"
    );
}

/// `-1` padding is what a row shorter than the budget answers, and it is the
/// half of the contract the ranking test above cannot reach.
///
/// `Toy::prefill`'s shortest request holds 3 keys and `TOP_K` is 3, so every
/// row there fills its budget. At a budget of 6 the two short rows cannot,
/// and what they must write is `0..nkeys` then `-1` — the kernel's
/// `nkeys <= topk` early-out, which is also the ONLY path that answers
/// without ranking anything.
#[test]
fn a_row_shorter_than_the_budget_pads_with_minus_one() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("index.topk") {
        return;
    }
    const WIDE_K: i32 = 6;
    let toy = Toy::prefill();
    let mut rng = Rng(0x1122_3344_5566_7788);
    let keys: Vec<u16> = (0..toy.key_elems()).map(|_| rng.bf16(1.0)).collect();
    let pool = Pool::build(&toy, &keys, true);

    let rows = toy.rows() as usize;
    let d = toy.head_dim as usize;
    let q: Vec<u16> = (0..rows * HEADS as usize * d)
        .map(|_| rng.bf16(1.0))
        .collect();
    let w: Vec<u16> = (0..rows * HEADS as usize).map(|_| rng.bf16(1.0)).collect();
    let d_q = Slab::of(&bytes_of_u16(&q));
    let d_w = Slab::of(&bytes_of_u16(&w));
    let d_sel = Slab::zeroed(rows * WIDE_K as usize * 4);

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    Index::topk::<bf16>(
        &ctx,
        In {
            ptr: d_q.ptr.cast(),
            rows: toy.rows(),
            width: HEADS * toy.head_dim,
        },
        In {
            ptr: d_w.ptr.cast(),
            rows: toy.rows(),
            width: HEADS,
        },
        pool.row(),
        HEADS.unsigned_abs(),
        toy.head_dim.unsigned_abs(),
        WIDE_K.unsigned_abs(),
        Out {
            ptr: d_sel.ptr.cast(),
            rows: toy.rows(),
            width: WIDE_K,
        },
    )
    .expect("the claimed `index.topk` body");
    let got = d_sel.read_i32(rows * WIDE_K as usize);

    let mut want = Vec::new();
    let mut short = 0;
    for t in 0..toy.rows() {
        let (_, abs) = toy.absolute(t);
        let nkeys = abs + 1;
        if nkeys < WIDE_K {
            short += 1;
        }
        want.extend(chosen(
            &logits(&toy, &keys, &q, &w, t, true, true),
            WIDE_K,
        ));
    }
    assert!(short > 0, "the toy has no row short of the budget to check");
    eprintln!("index.topk (budget {WIDE_K}): got {got:?}");
    assert_eq!(got, want, "a short row must answer `0..nkeys` then `-1`");
    assert!(
        got.contains(&-1),
        "a row shorter than the budget wrote no padding at all"
    );
}

// ── the operand column, without firing ───────────────────────────────────

/// What the two index bodies derive off the pool row, checked by what they
/// refuse.
///
/// Every check here refuses BEFORE anything reaches the device — the view is
/// read on the HOST, which is the whole of what a `Cache` mark carries — so
/// a host struct with dangling device pointers is exactly what these paths
/// see.
#[test]
fn the_index_column_refuses_what_it_cannot_derive() {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let toy = Toy::prefill();
    let blank = |seq_stride: i64, layout: i32, qo: *const i32, budget: i32| PagedKvView {
        keys: core::ptr::dangling_mut(),
        values: core::ptr::null_mut(),
        bf16_keys: core::ptr::dangling_mut(),
        bf16_values: core::ptr::null_mut(),
        page_indices: core::ptr::dangling(),
        page_indptr: core::ptr::dangling(),
        last_page_lens: core::ptr::dangling(),
        key_scales: core::ptr::null(),
        value_scales: core::ptr::null(),
        write_page: core::ptr::null(),
        write_offset: core::ptr::null(),
        page_size: toy.page_size,
        seq_stride,
        head_stride: seq_stride,
        layout,
        storage_dtype: kernels_cuda::attn::KvDType::Bf16 as i32,
        scheme_byte: kernels_cuda::attn::KvScheme::Native as i32,
        native_bf16: true,
        has_envelopes: false,
        env_min: core::ptr::null(),
        env_max: core::ptr::null(),
        block_size: 0,
        max_pages_per_request: budget,
        pages_in_batch: 1,
        qo_indptr: qo,
        row_valid: core::ptr::null(),
        requests: toy.requests(),
    };
    let csr: *const i32 = core::ptr::dangling();
    let row = |v: &PagedKvView| Cache::<Struct<KvCache>> {
        ptr: core::ptr::from_ref(v),
    };
    let k = |width: i32| In::<kernels_cuda::jit::abi::Tensor<bf16>> {
        ptr: core::ptr::dangling(),
        rows: toy.rows(),
        width,
    };

    // A pool whose token pitch is not the row being appended.
    let narrow = blank(i64::from(toy.head_dim) * 2, 0, csr, toy.pool_pages);
    let got = Index::kv_append::<bf16>(&ctx, k(toy.head_dim), row(&narrow));
    assert!(
        matches!(got, Err(Refusal::Narrow { .. })),
        "a pool whose stride disagrees with the row must be refused, not written: {got:?}"
    );

    // An HND pool: a token step there is one head wide and the row would
    // have to be scattered.
    let hnd = blank(i64::from(toy.head_dim), 1, csr, toy.pool_pages);
    let got = Index::kv_append::<bf16>(&ctx, k(toy.head_dim), row(&hnd));
    assert!(
        matches!(got, Err(Refusal::Absent { .. })),
        "an HND index pool must be named, not written sideways: {got:?}"
    );

    // A fire whose pool row carries no query CSR: the destination cannot be
    // resolved and the null is named rather than dereferenced.
    let no_csr = blank(i64::from(toy.head_dim), 0, core::ptr::null(), toy.pool_pages);
    let got = Index::kv_append::<bf16>(&ctx, k(toy.head_dim), row(&no_csr));
    assert!(
        matches!(got, Err(Refusal::Null { .. })),
        "a pool row with no query CSR must be named: {got:?}"
    );

    // A selection the statement allocated at a width other than the budget
    // it stated. The kernel writes `top_k` per row and would run off the
    // rectangle.
    let ok = blank(i64::from(toy.head_dim), 0, csr, toy.pool_pages);
    let sel = |width: i32| Out::<kernels_cuda::jit::abi::Tensor<i32>> {
        ptr: core::ptr::dangling_mut(),
        rows: toy.rows(),
        width,
    };
    let topk = |view: &PagedKvView, width: i32, budget: u32| {
        Index::topk::<bf16>(
            &ctx,
            In::<kernels_cuda::jit::abi::Tensor<bf16>> {
                ptr: core::ptr::dangling(),
                rows: toy.rows(),
                width: HEADS * toy.head_dim,
            },
            In::<kernels_cuda::jit::abi::Tensor<bf16>> {
                ptr: core::ptr::dangling(),
                rows: toy.rows(),
                width: HEADS,
            },
            Cache::<Struct<KvCache>> {
                ptr: core::ptr::from_ref(view),
            },
            HEADS.unsigned_abs(),
            toy.head_dim.unsigned_abs(),
            budget,
            sel(width),
        )
    };
    let got = topk(&ok, TOP_K + 1, TOP_K.unsigned_abs());
    assert!(
        matches!(got, Err(Refusal::Narrow { .. })),
        "a selection wider than the stated budget must be refused: {got:?}"
    );

    // A fire whose pool row states no page budget: the logits scratch has no
    // kv bound to be sized from, and it is named rather than guessed at.
    let unbounded = blank(i64::from(toy.head_dim), 0, csr, 0);
    let got = topk(&unbounded, TOP_K, TOP_K.unsigned_abs());
    assert!(
        matches!(got, Err(Refusal::Empty { .. })),
        "a pool row with no page budget must be named: {got:?}"
    );
}

/// What the selected attention derives off its own column, checked by what
/// it refuses — `mla_paged.rs`'s reading of the unselected pair, at the two
/// things a selection adds.
///
/// Both refuse BEFORE the device: the pool row is read on the host and every
/// check below is arithmetic over rectangles.
#[test]
fn the_selected_column_refuses_what_it_cannot_derive() {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let toy = Toy::decode();
    let view = PagedKvView {
        keys: core::ptr::dangling_mut(),
        values: core::ptr::dangling_mut(),
        bf16_keys: core::ptr::dangling_mut(),
        bf16_values: core::ptr::dangling_mut(),
        page_indices: core::ptr::dangling(),
        page_indptr: core::ptr::dangling(),
        last_page_lens: core::ptr::dangling(),
        key_scales: core::ptr::null(),
        value_scales: core::ptr::null(),
        write_page: core::ptr::null(),
        write_offset: core::ptr::null(),
        page_size: toy.page_size,
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
        max_pages_per_request: toy.pool_pages,
        pages_in_batch: 1,
        qo_indptr: core::ptr::dangling(),
        row_valid: core::ptr::null(),
        requests: toy.requests(),
    };
    let decode = |q_pe_width: i32, sel_rows: i32| {
        Mla::attention_decode_selected::<bf16>(
            &ctx,
            In::<kernels_cuda::jit::abi::Tensor<bf16>> {
                ptr: core::ptr::dangling(),
                rows: toy.rows(),
                width: MLA_HEADS * CKV,
            },
            In::<kernels_cuda::jit::abi::Tensor<bf16>> {
                ptr: core::ptr::dangling(),
                rows: toy.rows(),
                width: q_pe_width,
            },
            In::<kernels_cuda::jit::abi::Tensor<i32>> {
                ptr: core::ptr::dangling(),
                rows: sel_rows,
                width: 2,
            },
            Cache::<Struct<KvCache>> {
                ptr: core::ptr::from_ref(&view),
            },
            MLA_HEADS.unsigned_abs(),
            CKV.unsigned_abs(),
            0.5,
            Out::<kernels_cuda::jit::abi::Tensor<bf16>> {
                ptr: core::ptr::dangling_mut(),
                rows: toy.rows(),
                width: MLA_HEADS * CKV,
            },
        )
    };

    // A rotated half that does not divide by the stated head count — the
    // unselected pair's own refusal, which the selected one inherits by
    // reading the rope width the same way.
    let ragged = decode(MLA_HEADS * KPE - 1, toy.rows());
    assert!(
        matches!(ragged, Err(Refusal::Narrow { .. })),
        "a ragged rotated half must be refused, not divided: {ragged:?}"
    );

    // A selection with fewer rows than the query. The kernel reads
    // `selection + t * top_k` for every output row, so this is a read past
    // the rectangle rather than a wrong answer.
    let short = decode(MLA_HEADS * KPE, toy.rows() - 1);
    assert!(
        matches!(short, Err(Refusal::Narrow { .. })),
        "a selection shorter than the query must be refused: {short:?}"
    );
}

// ── which arm a selected fire takes ──────────────────────────────────────

/// A SELECTED FIRE TAKES THE SCALAR ARM, AT GLM'S OWN GEOMETRY — checked on
/// the host, because it is a property of `mla_naive::plan` and not of any
/// device.
///
/// glm5-a12b is `kv_lora_rank = 512`, `qk_rope_head_dim = 64`,
/// `heads = 96`, which is exactly `mma_supported`'s triple: an UNSELECTED
/// fire at that shape takes the tensor-core kernel. That kernel stages `kBK`
/// contiguous keys through one `cp.async` copy of `sK` and cannot walk an
/// index list, so `plan` declines it whenever a selection is present. This
/// asserts the branch both ways.
///
/// The scalar arm's shared memory at that shape is `8 * 512 + 16` floats =
/// 16 448 B, which is under the 48 KiB every CUDA device gives a block
/// without an opt-in — so glm's selected latent attention is runnable on
/// hardware where the UNSELECTED pair is not (there the only FA2 arm an
/// L40S-class part can pick writes past its own `SharedStorage` and is
/// refused by name).
#[test]
fn a_selected_fire_declines_the_tensor_core_arm() {
    use kernels_cuda::attn::mla_naive::{naive_smem_bytes, plan, NaivePlan, NaiveShape};

    let glm = NaiveShape {
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
        page_size: 16,
        total_tokens: 8,
        num_requests: 2,
        num_heads: 96,
        sm_scale: 0.5,
        causal: true,
        top_k: 0,
    };
    assert!(
        matches!(plan(glm, true, false), NaivePlan::Mma { .. }),
        "glm's shape is `mma_supported`'s triple; an unselected fire takes it"
    );
    assert!(
        matches!(plan(glm, true, true), NaivePlan::Scalar { .. }),
        "a selected fire must decline the tensor-core arm, which cannot walk a list"
    );
    assert_eq!(
        naive_smem_bytes(512),
        16_448,
        "the scalar arm's shared memory at glm's latent rank"
    );
    assert!(
        naive_smem_bytes(512) < 48 * 1024,
        "the scalar arm needs a shared-memory opt-in the launch does not make"
    );
}

/// WALKING A FULL SELECTION IS ATTENDING EVERY KEY, at glm's own geometry —
/// which is the equivalence that says the selection loop did not change what
/// a dense fire means, and the only thing in this tree that fires
/// `mla_mma_paged_kernel` at all.
///
/// Both arms lost a parameter to this work: the tensor-core kernel's dead
/// `index_mask`/`index_mask_stride` pair is deleted (nothing ever bound one)
/// and the scalar kernel's is replaced by the selection. `mla_paged.rs`
/// cannot reach either — on an L40S-class part its FA2 arm is the broken one
/// and it skips — so without this the shortened argument lists would compile
/// and never be launched.
///
/// The two arms disagree only by summation order and bf16 rounding, so the
/// bar is `mla_paged.rs`'s.
#[test]
fn a_full_selection_is_the_dense_attention() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("the latent attention arms at glm's geometry") {
        return;
    }
    use kernels_cuda::attn::mla_naive::{fire, MlaNaive, NaivePtrs, NaiveShape};

    // glm5-a12b's latent geometry, which is `mma_supported`'s triple.
    const G_CKV: i32 = 512;
    const G_KPE: i32 = 64;
    const G_HEADS: i32 = 96;
    const PAGE: i32 = 16;
    const KV_LEN: i32 = 20;
    let pages = (KV_LEN + PAGE - 1) / PAGE;
    let pool_pages = pages + 1;

    let mut rng = Rng(0xfeed_face_c0ff_ee11);
    let ckv: Vec<u16> = (0..(pool_pages * PAGE * G_CKV) as usize)
        .map(|_| rng.bf16(1.0))
        .collect();
    let kpe: Vec<u16> = (0..(pool_pages * PAGE * G_KPE) as usize)
        .map(|_| rng.bf16(1.0))
        .collect();
    let q: Vec<u16> = (0..(G_HEADS * G_CKV) as usize)
        .map(|_| rng.bf16(0.25))
        .collect();
    let q_pe: Vec<u16> = (0..(G_HEADS * G_KPE) as usize)
        .map(|_| rng.bf16(0.25))
        .collect();
    // The request's pages, scattered through the pool rather than `0..n`.
    let indices: Vec<u32> = (0..pages as u32).map(|p| pool_pages as u32 - 1 - p).collect();
    // The whole selection, ascending: every causal key, named.
    let selection: Vec<i32> = (0..KV_LEN).collect();

    let d_ckv = Slab::of(&bytes_of_u16(&ckv));
    let d_kpe = Slab::of(&bytes_of_u16(&kpe));
    let d_q = Slab::of(&bytes_of_u16(&q));
    let d_qpe = Slab::of(&bytes_of_u16(&q_pe));
    let d_idx = Slab::of(&bytes_of_u32(&indices));
    let d_indptr = Slab::of(&bytes_of_u32(&[0, pages as u32]));
    let d_lens = Slab::of(&bytes_of_u32(&[(KV_LEN - (pages - 1) * PAGE) as u32]));
    let d_qo = Slab::of(&bytes_of_u32(&[0, 1]));
    let d_sel = Slab::of(
        &selection
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let out_elems = (G_HEADS * G_CKV) as usize;
    let d_dense = Slab::zeroed(out_elems * 2);
    let d_walked = Slab::zeroed(out_elems * 2);

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let ptrs = |o: *mut c_void, sel: *const i32| NaivePtrs {
        q_nope: d_q.ptr.cast(),
        q_pe: d_qpe.ptr.cast(),
        ckv_pages: d_ckv.ptr.cast(),
        kpe_pages: d_kpe.ptr.cast(),
        qo_indptr: d_qo.ptr.cast(),
        kv_page_indices: d_idx.ptr.cast(),
        kv_page_indptr: d_indptr.ptr.cast(),
        kv_last_page_lens: d_lens.ptr.cast(),
        o: o.cast(),
        selection: sel,
    };
    let shape = |top_k: i32| NaiveShape {
        kv_lora_rank: G_CKV,
        qk_rope_head_dim: G_KPE,
        page_size: PAGE,
        total_tokens: 1,
        num_requests: 1,
        num_heads: G_HEADS,
        sm_scale: 1.0 / ((G_CKV + G_KPE) as f32).sqrt(),
        causal: false,
        top_k,
    };

    let dense = fire(&ctx, ptrs(d_dense.ptr, core::ptr::null()), shape(0));
    match dense {
        Ok(MlaNaive::LaunchedMma) => {}
        Ok(other) => panic!(
            "glm's shape must take the tensor-core arm when nothing selects: {}",
            match other {
                MlaNaive::LaunchedScalar => "it took the scalar one",
                MlaNaive::Declined(_) => "it declined",
                MlaNaive::LaunchedMma => unreachable!(),
            }
        ),
        Err(e) => {
            // The arm needs a ~100 KB shared-memory opt-in; a device that
            // will not grant one has nothing to compare against and says so.
            eprintln!("skipping the dense arm: {e:?}");
            return;
        }
    }
    let walked = fire(&ctx, ptrs(d_walked.ptr, d_sel.ptr.cast()), shape(KV_LEN))
        .expect("the scalar arm, walking a full selection");
    assert!(
        matches!(walked, MlaNaive::LaunchedScalar),
        "a selected fire must take the scalar arm"
    );
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "one of the two arms did not complete"
    );

    let a = d_dense.read_u16(out_elems);
    let b = d_walked.read_u16(out_elems);
    let want: Vec<f32> = a.iter().map(|x| wide(*x)).collect();
    let rms = agrees("a full selection vs the dense tensor-core arm", &b, &want);
    assert!(
        rms < 2e-2,
        "walking every key disagrees with attending every key ({rms:.3e})"
    );
    // AND THE COMPARISON IS NOT VACUOUS: two buffers of zeros would pass a
    // relative-rms bar with nothing in them.
    assert!(
        a.iter().any(|x| wide(*x) != 0.0),
        "the dense arm wrote nothing at all"
    );
}

// ── the selected attentions ──────────────────────────────────────────────

/// The latent geometry the scalar naive kernel can lane-split: both widths a
/// multiple of 32, the latent at most 512 and the rope at most 128.
const CKV: i32 = 32;
const KPE: i32 = 32;
const MLA_HEADS: i32 = 4;

/// A latent pool and the selection to attend it through.
struct Latent {
    _ckv: Slab,
    _kpe: Slab,
    _indices: Slab,
    _indptr: Slab,
    _lens: Slab,
    _qo: Slab,
    view: PagedKvView,
    h_ckv: Vec<u16>,
    h_kpe: Vec<u16>,
}

impl Latent {
    fn build(toy: &Toy, rng: &mut Rng) -> Latent {
        let slots = (toy.pool_pages * toy.page_size) as usize;
        let h_ckv: Vec<u16> = (0..slots * CKV as usize).map(|_| rng.bf16(1.0)).collect();
        let h_kpe: Vec<u16> = (0..slots * KPE as usize).map(|_| rng.bf16(1.0)).collect();
        let ckv = Slab::of(&bytes_of_u16(&h_ckv));
        let kpe = Slab::of(&bytes_of_u16(&h_kpe));
        let indices = Slab::of(&bytes_of_u32(&toy.page_indices));
        let indptr = Slab::of(&bytes_of_u32(&toy.page_indptr));
        let lens = Slab::of(&bytes_of_u32(&toy.last_page_lens));
        let qo = Slab::of(&bytes_of_u32(&toy.qo_indptr));
        let view = PagedKvView {
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
            // view's; the statement's operands carry both.
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
            max_pages_per_request: toy.pool_pages,
            pages_in_batch: toy.page_indices.len() as i32,
            qo_indptr: qo.ptr.cast(),
            row_valid: core::ptr::null(),
            requests: toy.requests(),
        };
        Latent {
            _ckv: ckv,
            _kpe: kpe,
            _indices: indices,
            _indptr: indptr,
            _lens: lens,
            _qo: qo,
            view,
            h_ckv,
            h_kpe,
        }
    }

    fn row(&self) -> Cache<Struct<KvCache>> {
        Cache {
            ptr: core::ptr::from_ref(&self.view),
        }
    }
}

/// One softmax per head over the keys `walk` names, on the host.
///
/// `walk(t)` is the key list for row `t` — the selection for the reading
/// under test, every causal key for the mutation.
fn attended(
    toy: &Toy,
    lat: &Latent,
    q: &[u16],
    q_pe: &[u16],
    sm_scale: f32,
    walk: impl Fn(i32) -> Vec<i32>,
) -> Vec<f32> {
    let rows = toy.rows() as usize;
    let mut out = vec![0.0f32; rows * MLA_HEADS as usize * CKV as usize];
    for t in 0..toy.rows() {
        let (r, _) = toy.absolute(t);
        let keys = walk(t);
        for h in 0..MLA_HEADS as usize {
            let qn = &q[(t as usize * MLA_HEADS as usize + h) * CKV as usize..][..CKV as usize];
            let qp = &q_pe[(t as usize * MLA_HEADS as usize + h) * KPE as usize..][..KPE as usize];
            let mut scores = Vec::with_capacity(keys.len());
            for j in &keys {
                let slot = toy.slot(r, *j);
                let c0 = slot * CKV as usize;
                let p0 = slot * KPE as usize;
                let mut s = 0.0f32;
                for i in 0..CKV as usize {
                    s += wide(qn[i]) * wide(lat.h_ckv[c0 + i]);
                }
                for i in 0..KPE as usize {
                    s += wide(qp[i]) * wide(lat.h_kpe[p0 + i]);
                }
                scores.push(s * sm_scale);
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                denom += *s;
            }
            let at = (t as usize * MLA_HEADS as usize + h) * CKV as usize;
            for (n, j) in keys.iter().enumerate() {
                let c0 = toy.slot(r, *j) * CKV as usize;
                let p = scores[n] / denom;
                for i in 0..CKV as usize {
                    out[at + i] += p * wide(lat.h_ckv[c0 + i]);
                }
            }
        }
    }
    out
}

/// The relative rms of `got` against `want`, at `mla_paged.rs`'s bar.
fn agrees(what: &str, got: &[u16], want: &[f32]) -> f32 {
    let (mut num, mut den, mut worst) = (0.0f64, 0.0f64, 0.0f32);
    for (g, w) in got.iter().zip(want.iter()) {
        let e = wide(*g) - *w;
        num += f64::from(e) * f64::from(e);
        den += f64::from(*w) * f64::from(*w);
        worst = worst.max(e.abs());
    }
    let rms = (num / den.max(1e-30)).sqrt() as f32;
    eprintln!("{what}: relative rms {rms:.3e}, worst |err| {worst:.3e}");
    rms
}

/// The selection each row of the toy attends: a deterministic spread of the
/// causal range, so the reference and the fire walk the same keys and the
/// "every key" mutation is a different set.
fn spread(toy: &Toy, t: i32, top_k: i32) -> Vec<i32> {
    let (_, abs) = toy.absolute(t);
    let nkeys = abs + 1;
    let mut keep: Vec<i32> = (0..nkeys).rev().step_by(2).take(top_k as usize).collect();
    keep.sort_unstable();
    keep
}

fn selection_rows(toy: &Toy, top_k: i32) -> Vec<i32> {
    let mut sel = Vec::new();
    for t in 0..toy.rows() {
        let mut row = spread(toy, t, top_k);
        row.resize(top_k as usize, -1);
        sel.extend(row);
    }
    sel
}

#[test]
fn the_prefill_attends_only_the_selected_keys() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("mla.attention_prefill_selected") {
        return;
    }
    const SEL_K: i32 = 2;
    let toy = Toy::prefill();
    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    let lat = Latent::build(&toy, &mut rng);
    let rows = toy.rows() as usize;
    let q: Vec<u16> = (0..rows * (MLA_HEADS * CKV) as usize)
        .map(|_| rng.bf16(0.5))
        .collect();
    let q_pe: Vec<u16> = (0..rows * (MLA_HEADS * KPE) as usize)
        .map(|_| rng.bf16(0.5))
        .collect();
    let sm_scale = 1.0f32 / ((CKV + KPE) as f32).sqrt();
    let sel = selection_rows(&toy, SEL_K);

    let d_q = Slab::of(&bytes_of_u16(&q));
    let d_qpe = Slab::of(&bytes_of_u16(&q_pe));
    let d_sel = Slab::of(&sel.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let d_indptr = Slab::of(&bytes_of_u32(&toy.qo_indptr));
    let d_out = Slab::zeroed(rows * (MLA_HEADS * CKV) as usize * 2);

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    Mla::attention_prefill_selected::<bf16>(
        &ctx,
        In {
            ptr: d_q.ptr.cast(),
            rows: toy.rows(),
            width: MLA_HEADS * CKV,
        },
        In {
            ptr: d_indptr.ptr.cast(),
            rows: toy.requests(),
            width: 1,
        },
        In {
            ptr: d_qpe.ptr.cast(),
            rows: toy.rows(),
            width: MLA_HEADS * KPE,
        },
        In {
            ptr: d_sel.ptr.cast(),
            rows: toy.rows(),
            width: SEL_K,
        },
        lat.row(),
        MLA_HEADS.unsigned_abs(),
        CKV.unsigned_abs(),
        sm_scale,
        Out {
            ptr: d_out.ptr.cast(),
            rows: toy.rows(),
            width: MLA_HEADS * CKV,
        },
    )
    .expect("the claimed `mla.attention_prefill_selected` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the selected prefill did not complete"
    );
    let got = d_out.read_u16(rows * (MLA_HEADS * CKV) as usize);

    let want = attended(&toy, &lat, &q, &q_pe, sm_scale, |t| spread(&toy, t, SEL_K));
    // THE BAR IS BF16'S, `mla_paged.rs`'s for the same kernel: the reference
    // sums in index order and the kernel merges eight per-warp softmax
    // states before storing bf16.
    let rms = agrees("mla.attention_prefill_selected", &got, &want);
    assert!(rms < 2e-2, "relative rms {rms:.3e} against the host softmax");

    // THE MUTATION: attending the whole causal range instead of the
    // selection. A kernel that ignored the list would land here.
    let dense = attended(&toy, &lat, &q, &q_pe, sm_scale, |t| {
        let (_, abs) = toy.absolute(t);
        (0..=abs).collect()
    });
    let loose = agrees("mla.attention_prefill_selected (dense)", &got, &dense);
    assert!(
        loose > 1e-1,
        "attending every causal key answers the same rows ({loose:.3e}) — the \
         selection is not being walked"
    );
}

#[test]
fn the_decode_attends_only_the_selected_keys() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("mla.attention_decode_selected") {
        return;
    }
    const SEL_K: i32 = 2;
    let toy = Toy::decode();
    let mut rng = Rng(0x0f0f_5a5a_9696_3c3c);
    let lat = Latent::build(&toy, &mut rng);
    let rows = toy.rows() as usize;
    let q: Vec<u16> = (0..rows * (MLA_HEADS * CKV) as usize)
        .map(|_| rng.bf16(0.5))
        .collect();
    let q_pe: Vec<u16> = (0..rows * (MLA_HEADS * KPE) as usize)
        .map(|_| rng.bf16(0.5))
        .collect();
    let sm_scale = 1.0f32 / ((CKV + KPE) as f32).sqrt();
    // A DECODE ROW SEES THE WHOLE PREFIX, so its selection may name any
    // cached key — the point states no causal order and the naive kernel is
    // fired with `causal = false`. `spread` still ranges over `abs + 1`,
    // which for a decode row IS the whole cache.
    let sel = selection_rows(&toy, SEL_K);

    let d_q = Slab::of(&bytes_of_u16(&q));
    let d_qpe = Slab::of(&bytes_of_u16(&q_pe));
    let d_sel = Slab::of(&sel.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let d_out = Slab::zeroed(rows * (MLA_HEADS * CKV) as usize * 2);

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    Mla::attention_decode_selected::<bf16>(
        &ctx,
        In {
            ptr: d_q.ptr.cast(),
            rows: toy.rows(),
            width: MLA_HEADS * CKV,
        },
        In {
            ptr: d_qpe.ptr.cast(),
            rows: toy.rows(),
            width: MLA_HEADS * KPE,
        },
        In {
            ptr: d_sel.ptr.cast(),
            rows: toy.rows(),
            width: SEL_K,
        },
        lat.row(),
        MLA_HEADS.unsigned_abs(),
        CKV.unsigned_abs(),
        sm_scale,
        Out {
            ptr: d_out.ptr.cast(),
            rows: toy.rows(),
            width: MLA_HEADS * CKV,
        },
    )
    .expect("the claimed `mla.attention_decode_selected` body");
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "the selected decode did not complete"
    );
    let got = d_out.read_u16(rows * (MLA_HEADS * CKV) as usize);

    let want = attended(&toy, &lat, &q, &q_pe, sm_scale, |t| spread(&toy, t, SEL_K));
    let rms = agrees("mla.attention_decode_selected", &got, &want);
    assert!(rms < 2e-2, "relative rms {rms:.3e} against the host softmax");

    // THE MUTATION: the whole cached prefix instead of the selection.
    let dense = attended(&toy, &lat, &q, &q_pe, sm_scale, |t| {
        let (r, _) = toy.absolute(t);
        (0..toy.kv_len(r)).collect()
    });
    let loose = agrees("mla.attention_decode_selected (dense)", &got, &dense);
    assert!(
        loose > 1e-1,
        "attending the whole prefix answers the same rows ({loose:.3e}) — the \
         selection is not being walked"
    );
}
