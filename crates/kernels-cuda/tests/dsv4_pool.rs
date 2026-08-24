//! The `Pool` family as CLAIMED BODIES — deepseek-v4's compressed plane.
//!
//! All five points were claim-only until R4b, and each for the same kind of
//! reason: the launcher reads objects a statement cannot name. Three
//! RESIDENTS beside the page table (the state halves, the running scores, the
//! absolute-position table), the compressed pool itself, and two runtime
//! planes the fire stages (`row_valid`, `request_of_token`). Every one of
//! those is a `Raise` with a key, so what closed them is the door on the
//! plane's own context: `Ctx::raised::<R>()` / `Ctx::staged::<R>()`, answered
//! here by a hand-built [`Staged`] exactly as `driver-cuda`'s `FireViews`
//! answers it in a fire.
//!
//! # This is the only place these bodies run
//!
//! `driver-cuda` stages NONE of the dsv4 objects — they are on its own
//! `UNSTAGED` list — so a dsv4 lane still refuses there, by key. That is the
//! honest state and it is why these tests matter more than usual: they are
//! the only execution the four bodies get, and the references below are
//! transcriptions of `attn/dsv4_compress.cuh` and of nothing else.
//!
//! # What each test can see
//!
//! * **The boundary meta's THIRD result.** The kernels write `out_rope`
//!   unconditionally and no text reads it, so the body sinks it into plane
//!   scratch. A body that aimed it at a stated rectangle instead would
//!   clobber `boundary_req`, which the first two tests would catch, and a
//!   body that passed null would fault.
//! * **The row-validity plane, through the OPTIONAL door.** These two points
//!   name no cache row, so they cannot read `row_valid` off a pool view the
//!   way every appender does; they ask for `"row_valid"` by key. A rejected
//!   row must not close a window.
//! * **The compressor's window multiplier.** `coff` is derived from the
//!   stated ratio rather than stated — the declaration decided that — and at
//!   ratio 4 it is 2, so the gather pools over EIGHT tokens and not four.
//!   `the_gather_pools_the_whole_window` is the row that says so.
//! * **The compressed append lands where the boundary says**, through the
//!   page table the statement names and the pool plane the key answers.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Pool as PoolPoints;
use kernels::raises::{Answered, Struct};
use kernels::routine::{Cache, In, Out};
use kernels::Refusal;
use kernels_cuda::jit::abi::{bf16, Tensor};
use kernels_cuda::jit::Ctx;
use kernels_cuda::views::{KvCache, PagedKvView};

/// The device scratch is a process-global named-slab arena sized for one fire
/// at a time, and the boundary bodies take a slab out of it. The lock is
/// `gdn_chunk_prefill.rs`'s, and here it is load-bearing rather than
/// precautionary.
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

// ── device memory ────────────────────────────────────────────────────────

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

    fn read_i32(&self, elems: usize) -> Vec<i32> {
        self.read(elems * 4)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_u16(&self, elems: usize) -> Vec<u16> {
        self.read(elems * 2)
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

fn narrow(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

fn widen(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

// ── the fire this family sees ────────────────────────────────────────────

/// Two requests, a scattered page table, a rejected row, and a ratio whose
/// `coff` is not one.
struct Toy {
    ratio: i32,
    head_dim: i32,
    page_size: i32,
    pool_pages: i32,
    /// One absolute position per token row.
    positions: Vec<i32>,
    /// The query CSR, `[requests + 1]`.
    qo_indptr: Vec<u32>,
    page_indptr: Vec<u32>,
    page_indices: Vec<u32>,
    row_valid: Vec<u8>,
}

impl Toy {
    fn small() -> Toy {
        Toy {
            // RATIO 4 IS THE ONE WHOSE `coff` IS 2, which is the only value
            // that separates a body deriving the multiplier from one that
            // assumed a window of `ratio`.
            ratio: 4,
            head_dim: 8,
            page_size: 4,
            pool_pages: 8,
            // Row 2 closes a window (`(11 + 1) % 4 == 0`) and is REJECTED;
            // rows 1 and 5 close one and are valid; the rest do not.
            positions: vec![6, 7, 11, 12, 14, 15],
            qo_indptr: vec![0, 3, 6],
            // ENOUGH PAGES FOR EVERY POSITION EACH REQUEST HOLDS, not just
            // for its new rows: the gather walks back `coff * ratio` tokens
            // from a boundary and addresses every one of them through this
            // table. Request 0 reaches position 11 (three pages), request 1
            // reaches 15 (four).
            page_indptr: vec![0, 3, 7],
            page_indices: vec![5, 2, 7, 1, 6, 3, 4],
            row_valid: vec![1, 1, 0, 1, 1, 1],
        }
    }

    fn rows(&self) -> i32 {
        self.positions.len() as i32
    }

    fn requests(&self) -> i32 {
        self.qo_indptr.len() as i32 - 1
    }

    fn coff(&self) -> i32 {
        if self.ratio == 4 { 2 } else { 1 }
    }

    fn width(&self) -> i32 {
        self.coff() * self.head_dim
    }

    fn slots(&self) -> usize {
        (self.pool_pages * self.page_size) as usize
    }

    /// `pie::attn::paged_slot`, on the host.
    fn slot(&self, req: i32, pos: i32) -> usize {
        let page =
            self.page_indices[(self.page_indptr[req as usize] as i32 + pos / self.page_size) as usize];
        (page as i32 * self.page_size + pos % self.page_size) as usize
    }

    /// Which request row `t` belongs to, by the CSR the prefill form walks.
    fn request_of(&self, t: i32) -> i32 {
        (0..self.requests())
            .rev()
            .find(|r| self.qo_indptr[*r as usize] as i32 <= t)
            .unwrap_or(0)
    }

    /// The boundary metadata, on the host: `pie::attn::dsv4_boundary_meta_*`.
    fn boundaries(&self, prefill: bool) -> (Vec<i32>, Vec<i32>) {
        let mut pos = Vec::new();
        let mut req = Vec::new();
        for t in 0..self.rows() {
            let p = self.positions[t as usize];
            let valid = self.row_valid[t as usize] != 0;
            let is_boundary = valid && ((p + 1) % self.ratio == 0);
            pos.push(if is_boundary { p } else { -1 });
            req.push(if prefill { self.request_of(t) } else { t });
        }
        (pos, req)
    }
}

/// The compressed plane's residents, staged the way a driver would have to.
struct Planes {
    state_kv: Slab,
    state_score: Slab,
    ape: Slab,
    comp_kv: Slab,
    row_valid: Slab,
    request_of_token: Slab,
    h_state_kv: Vec<u16>,
    h_state_score: Vec<u16>,
    h_ape: Vec<f32>,
}

impl Planes {
    fn build(toy: &Toy) -> Planes {
        let mut rng = Rng(0xfeed_face_1234_5678);
        let n = toy.slots() * toy.width() as usize;
        let h_state_kv: Vec<u16> = (0..n).map(|_| narrow(rng.next() * 2.0)).collect();
        let h_state_score: Vec<u16> = (0..n).map(|_| narrow(rng.next())).collect();
        let h_ape: Vec<f32> = (0..(toy.ratio * toy.width()) as usize)
            .map(|_| rng.next() * 0.5)
            .collect();
        let request_of_token: Vec<i32> = (0..toy.rows()).map(|t| toy.request_of(t)).collect();
        Planes {
            state_kv: Slab::of(&bytes_of_u16(&h_state_kv)),
            state_score: Slab::of(&bytes_of_u16(&h_state_score)),
            ape: Slab::of(&bytes_of_f32(&h_ape)),
            // A POISON FILL: an append that wrote nothing would pass a
            // zero-vs-zero comparison at every slot it should have skipped.
            comp_kv: Slab::of(&bytes_of_u16(&vec![
                narrow(-9.0);
                toy.slots() * toy.head_dim as usize
            ])),
            row_valid: Slab::of(&toy.row_valid),
            request_of_token: Slab::of(&bytes_of_i32(&request_of_token)),
            h_state_kv,
            h_state_score,
            h_ape,
        }
    }
}

/// What this test stages, by the key each object's `Raise` declares.
struct Staged {
    state_kv: *const c_void,
    state_score: *const c_void,
    ape: *const c_void,
    comp_kv: *const c_void,
    row_valid: *const c_void,
    request_of_token: *const c_void,
}

impl Answered for Staged {
    fn raised(&self, key: &'static str) -> Option<*const c_void> {
        match key {
            "dsv4.state_kv" => Some(self.state_kv),
            "dsv4.state_score" => Some(self.state_score),
            "dsv4.ape" => Some(self.ape),
            "dsv4.comp_kv_pages" => Some(self.comp_kv),
            "row_valid" => Some(self.row_valid),
            "request_of_token" => Some(self.request_of_token),
            _ => None,
        }
    }
}

impl Staged {
    fn of(p: &Planes) -> Staged {
        Staged {
            state_kv: p.state_kv.ptr.cast_const(),
            state_score: p.state_score.ptr.cast_const(),
            ape: p.ape.ptr.cast_const(),
            comp_kv: p.comp_kv.ptr.cast_const(),
            row_valid: p.row_valid.ptr.cast_const(),
            request_of_token: p.request_of_token.ptr.cast_const(),
        }
    }
}

/// The page table a statement names. Only the page CSR, the page indices and
/// the page size are read by this family's kernels.
struct Pages {
    _indices: Slab,
    _indptr: Slab,
    _lens: Slab,
    _qo: Slab,
    view: PagedKvView,
}

impl Pages {
    fn build(toy: &Toy) -> Pages {
        let indices = Slab::of(&bytes_of_u32(&toy.page_indices));
        let indptr = Slab::of(&bytes_of_u32(&toy.page_indptr));
        let lens = Slab::of(&bytes_of_u32(&vec![toy.page_size as u32; toy.requests() as usize]));
        let qo = Slab::of(&bytes_of_u32(&toy.qo_indptr));
        let view = PagedKvView {
            keys: core::ptr::null_mut(),
            values: core::ptr::null_mut(),
            bf16_keys: core::ptr::null_mut(),
            bf16_values: core::ptr::null_mut(),
            page_indices: indices.ptr.cast(),
            page_indptr: indptr.ptr.cast(),
            last_page_lens: lens.ptr.cast(),
            key_scales: core::ptr::null(),
            value_scales: core::ptr::null(),
            write_page: core::ptr::null(),
            write_offset: core::ptr::null(),
            page_size: toy.page_size,
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
            max_pages_per_request: toy.page_indices.len() as i32,
            pages_in_batch: toy.page_indices.len() as i32,
            qo_indptr: qo.ptr.cast(),
            row_valid: core::ptr::null(),
            requests: toy.requests(),
        };
        Pages {
            _indices: indices,
            _indptr: indptr,
            _lens: lens,
            _qo: qo,
            view,
        }
    }

    fn cache(&self) -> Cache<Struct<KvCache>> {
        Cache {
            ptr: core::ptr::from_ref(&self.view),
        }
    }
}

fn sync(what: &str) {
    assert_eq!(
        unsafe { rt::cudaDeviceSynchronize() },
        rt::cudaError::cudaSuccess,
        "{what} did not complete"
    );
}

// ── the boundary metadata ────────────────────────────────────────────────

/// The decode form: one token per request, so the request column is the row
/// index and nothing is searched.
#[test]
fn the_decode_boundaries_are_the_positions_that_close_a_window() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("pool.boundary_decode") {
        return;
    }
    let toy = Toy::small();
    let planes = Planes::build(&toy);
    let staged = Staged::of(&planes);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) }.with_raised(&staged);

    let d_pos = Slab::of(&bytes_of_i32(&toy.positions));
    // POISONED, so a result the body never wrote is visible.
    let out_pos = Slab::of(&bytes_of_i32(&vec![-99; toy.rows() as usize]));
    let out_req = Slab::of(&bytes_of_i32(&vec![-99; toy.rows() as usize]));
    let i32_in = |s: &Slab| In::<Tensor<i32>> {
        ptr: s.ptr.cast(),
        rows: toy.rows(),
        width: 1,
    };
    let i32_out = |s: &Slab| Out::<Tensor<i32>> {
        ptr: s.ptr.cast(),
        rows: toy.rows(),
        width: 1,
    };
    PoolPoints::boundary_decode(
        &ctx,
        i32_in(&d_pos),
        toy.ratio.unsigned_abs(),
        i32_out(&out_pos),
        i32_out(&out_req),
    )
    .expect("the claimed `pool.boundary_decode` body");
    sync("the boundary meta");

    let (want_pos, want_req) = toy.boundaries(false);
    assert_eq!(out_pos.read_i32(toy.rows() as usize), want_pos);
    assert_eq!(out_req.read_i32(toy.rows() as usize), want_req);
}

/// The prefill form: many rows per request, so the request column is the CSR
/// row each token index falls in.
#[test]
fn the_prefill_boundaries_read_the_request_off_the_csr() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("pool.boundary_prefill") {
        return;
    }
    let toy = Toy::small();
    let planes = Planes::build(&toy);
    let staged = Staged::of(&planes);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) }.with_raised(&staged);

    let d_pos = Slab::of(&bytes_of_i32(&toy.positions));
    let d_csr = Slab::of(&bytes_of_u32(&toy.qo_indptr));
    let out_pos = Slab::of(&bytes_of_i32(&vec![-99; toy.rows() as usize]));
    let out_req = Slab::of(&bytes_of_i32(&vec![-99; toy.rows() as usize]));
    PoolPoints::boundary_prefill(
        &ctx,
        In {
            ptr: d_pos.ptr.cast(),
            rows: toy.rows(),
            width: 1,
        },
        In {
            ptr: d_csr.ptr.cast(),
            // THE REQUEST COUNT IS THE CSR OPERAND'S OWN ROW COUNT, which is
            // what every prefill routine in this crate reads.
            rows: toy.requests(),
            width: 1,
        },
        toy.ratio.unsigned_abs(),
        Out {
            ptr: out_pos.ptr.cast(),
            rows: toy.rows(),
            width: 1,
        },
        Out {
            ptr: out_req.ptr.cast(),
            rows: toy.rows(),
            width: 1,
        },
    )
    .expect("the claimed `pool.boundary_prefill` body");
    sync("the boundary meta");

    let (want_pos, want_req) = toy.boundaries(true);
    assert_eq!(out_pos.read_i32(toy.rows() as usize), want_pos);
    assert_eq!(
        out_req.read_i32(toy.rows() as usize),
        want_req,
        "the request column is the CSR row each token falls in"
    );
}

// ── the gather ───────────────────────────────────────────────────────────

/// One pooled entry per boundary, over the `coff * ratio` tokens ending
/// there.
///
/// THE WINDOW IS EIGHT AND NOT FOUR at this ratio, which is the whole reason
/// `coff` exists and the whole reason the body derives it: the reference
/// below walks `coff * ratio` and a body that walked `ratio` disagrees at
/// every element of every entry.
#[test]
fn the_gather_pools_the_whole_window() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("pool.gather") {
        return;
    }
    let toy = Toy::small();
    let planes = Planes::build(&toy);
    let pages = Pages::build(&toy);
    let staged = Staged::of(&planes);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) }.with_raised(&staged);

    let (bpos, breq) = toy.boundaries(true);
    let d_bpos = Slab::of(&bytes_of_i32(&bpos));
    let d_breq = Slab::of(&bytes_of_i32(&breq));
    let entries = Slab::of(&bytes_of_u16(&vec![
        narrow(-9.0);
        (toy.rows() * toy.head_dim) as usize
    ]));
    PoolPoints::gather::<bf16>(
        &ctx,
        In {
            ptr: d_bpos.ptr.cast(),
            rows: toy.rows(),
            width: 1,
        },
        In {
            ptr: d_breq.ptr.cast(),
            rows: toy.rows(),
            width: 1,
        },
        pages.cache(),
        toy.head_dim.unsigned_abs(),
        toy.ratio.unsigned_abs(),
        Out {
            ptr: entries.ptr.cast(),
            rows: toy.rows(),
            width: toy.head_dim,
        },
    )
    .expect("the claimed `pool.gather` body");
    sync("the gather");

    // `pie::attn::dsv4_compress_gather_paged`, on the host.
    let window = toy.coff() * toy.ratio;
    let width = toy.width() as usize;
    let mut want = vec![0f32; (toy.rows() * toy.head_dim) as usize];
    for c in 0..toy.rows() as usize {
        if bpos[c] < 0 {
            continue;
        }
        for d in 0..toy.head_dim as usize {
            let score = |i: i32| -> Option<f32> {
                let pos = bpos[c] + i - (window - 1);
                if pos < 0 {
                    return None;
                }
                let col = if i >= toy.ratio { toy.head_dim as usize } else { 0 } + d;
                let slot = toy.slot(breq[c], pos);
                let mut s = widen(planes.h_state_score[slot * width + col]);
                s += planes.h_ape[(pos % toy.ratio) as usize * width + col];
                Some(s)
            };
            let mut max_s = f32::NEG_INFINITY;
            for i in 0..window {
                if let Some(s) = score(i) {
                    max_s = max_s.max(s);
                }
            }
            if !max_s.is_finite() {
                continue;
            }
            let (mut sum_e, mut acc) = (0f32, 0f32);
            for i in 0..window {
                let Some(s) = score(i) else { continue };
                let pos = bpos[c] + i - (window - 1);
                let col = if i >= toy.ratio { toy.head_dim as usize } else { 0 } + d;
                let slot = toy.slot(breq[c], pos);
                let e = (s - max_s).exp();
                sum_e += e;
                acc += e * widen(planes.h_state_kv[slot * width + col]);
            }
            want[c * toy.head_dim as usize + d] = if sum_e > 0.0 { acc / sum_e } else { 0.0 };
        }
    }

    let got = entries.read_u16((toy.rows() * toy.head_dim) as usize);
    // `__expf` IS THE FAST INTRINSIC and the accumulation order is the
    // kernel's, so this is a tolerance and not an equality — the separation
    // the test needs is against a FOUR-wide window, which moves entries by
    // whole units, not by ulps.
    let mut worst = 0f32;
    for (i, w) in want.iter().enumerate() {
        worst = worst.max((widen(got[i]) - w).abs());
    }
    eprintln!("pool.gather: worst |diff| {worst:.6} over {} element(s)", want.len());
    assert!(
        worst < 0.02,
        "the gather disagrees with the host pool by {worst}"
    );
    assert!(
        want.iter().any(|x| x.abs() > 0.01),
        "every entry is zero; the reference proves nothing"
    );
}

// ── the compressed append ────────────────────────────────────────────────

#[test]
fn the_compressed_append_lands_where_the_boundary_says() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("pool.kv_append") {
        return;
    }
    let toy = Toy::small();
    let planes = Planes::build(&toy);
    let pages = Pages::build(&toy);
    let staged = Staged::of(&planes);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) }.with_raised(&staged);

    let (bpos, breq) = toy.boundaries(true);
    let d_bpos = Slab::of(&bytes_of_i32(&bpos));
    let d_breq = Slab::of(&bytes_of_i32(&breq));
    let mut rng = Rng(0x0102_0304_0506_0708);
    let h_entries: Vec<u16> = (0..(toy.rows() * toy.head_dim) as usize)
        .map(|_| narrow(rng.next() * 4.0))
        .collect();
    let d_entries = Slab::of(&bytes_of_u16(&h_entries));

    PoolPoints::kv_append::<bf16>(
        &ctx,
        In {
            ptr: d_entries.ptr.cast(),
            rows: toy.rows(),
            width: toy.head_dim,
        },
        In {
            ptr: d_bpos.ptr.cast(),
            rows: toy.rows(),
            width: 1,
        },
        In {
            ptr: d_breq.ptr.cast(),
            rows: toy.rows(),
            width: 1,
        },
        pages.cache(),
    )
    .expect("the claimed `pool.kv_append` body");
    sync("the compressed append");

    let head = toy.head_dim as usize;
    let mut want = vec![narrow(-9.0); toy.slots() * head];
    for c in 0..toy.rows() as usize {
        if bpos[c] < 0 {
            continue;
        }
        let slot = toy.slot(breq[c], bpos[c]);
        want[slot * head..slot * head + head]
            .copy_from_slice(&h_entries[c * head..c * head + head]);
    }
    let got = planes.comp_kv.read_u16(toy.slots() * head);
    let bad = (0..got.len()).filter(|i| got[*i] != want[*i]).count();
    eprintln!(
        "pool.kv_append: {}/{} exact",
        got.len() - bad,
        got.len()
    );
    // A SCATTER IS EXACT OR IT IS WRONG.
    assert_eq!(bad, 0, "{bad} element(s) landed wrong in the compressed pool");
    assert!(
        want.iter().any(|x| *x != narrow(-9.0)),
        "nothing was appended; the reference proves nothing"
    );
}

// ── the door ─────────────────────────────────────────────────────────────

/// A context that stages nothing refuses BY THE KEY — which is what
/// `driver-cuda` does for every one of these today, and the sentence a reader
/// needs: the thing to build is named.
#[test]
fn an_unstaged_context_names_the_resident_it_wanted() {
    let toy = Toy::small();
    let pages = Pages::build(&toy);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let nothing_i32 = In::<Tensor<i32>> {
        ptr: core::ptr::null(),
        rows: toy.rows(),
        width: 1,
    };
    let refused = PoolPoints::gather::<bf16>(
        &ctx,
        nothing_i32,
        nothing_i32,
        pages.cache(),
        toy.head_dim.unsigned_abs(),
        toy.ratio.unsigned_abs(),
        Out {
            ptr: core::ptr::null_mut(),
            rows: toy.rows(),
            width: toy.head_dim,
        },
    );
    assert!(
        matches!(refused, Err(Refusal::Absent { what: "dsv4.state_kv" })),
        "an unstaged gather must name the resident, got {refused:?}"
    );
    let refused = PoolPoints::kv_append::<bf16>(
        &ctx,
        In::<Tensor<bf16>> {
            ptr: core::ptr::null(),
            rows: toy.rows(),
            width: toy.head_dim,
        },
        nothing_i32,
        nothing_i32,
        pages.cache(),
    );
    assert!(
        matches!(
            refused,
            Err(Refusal::Absent {
                what: "dsv4.comp_kv_pages"
            })
        ),
        "an unstaged compressed append must name the pool, got {refused:?}"
    );
}
