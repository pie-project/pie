//! The TIER-2 arm, fire-locked: `cuda::qkv_fused_qknorm_rope_vnorm_write`.
//!
//! # Why this test and not a smoke
//!
//! gemma is the SKU that states this point, and `baker-smoke --sku
//! gemma4-e4b-bf16-kv-bf16` fires it — five times, on real weights, through
//! the generated dispatch — before stopping at something else entirely:
//! `attention.decode` at head width 512 against a schedule planned at 256.
//! gemma-e4b's tower states TWO head widths (35 decode statements at 256,
//! 7 at 512) and an executor raises ONE fa2 decode schedule per fire, so
//! `agrees` refuses the second width by name. That refusal is R4b's named
//! seam, it is downstream of this point, and it means no ARGMAX comes back to
//! compare. So the lock on this arm is here instead.
//!
//! # What is locked
//!
//! The method against THE LAUNCH THE ROUTINE BUILT. `attn::qkv_fused`'s
//! `qkv_decode_qk_norm_rope_write_kv_bf16` and its
//! `qkv_decode_fused_dispatch` were deleted when this body took their
//! content; what stood in them is transcribed into [`by_name`] below, arg for
//! arg and in their order, and the two fires must leave BIT-IDENTICAL bytes
//! in the q rectangle and in both page planes. A body that derived
//! `num_q_heads` differently, chose the other instantiation, transposed two
//! of twenty-two arguments, or reached for the wrong plane of the pool row
//! fails that and passes nothing else.
//!
//! THE MUTATIONS ARE WHAT GIVE IT TEETH. A comparison of one kernel with
//! itself is vacuous unless the inputs it reads are shown to matter, so every
//! scalar this statement carries is perturbed on the method's side alone and
//! the answer must move: the epsilon, the head counts, the rope base and the
//! per-row position. Each is a column the generated arm reads by index, and
//! an index off by one is exactly what a mutation that does not move the
//! answer would be hiding.
//!
//! # The geometry is gemma's
//!
//! head_dim 256, 8 q heads, 2 kv heads, page 16 — the sliding layer of
//! `gemma4-e4b-bf16-kv-bf16`, whose 20 fused statements all state
//! `(kv_heads 2, head_dim 256)`. 256 is a WARP-ARM width, so this exercises
//! the arm the SKU actually takes; `the_block_arm_answers_a_width_the_warp_
//! arm_has_no_tiling_for` takes the other one.

#![cfg(feature = "_cuda")]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::raises::Struct;
use kernels::routine::{Cache, Const, In, Out, Refusal};
use kernels::{Bind, Fire};
use kernels_cuda::jit::abi::{bf16, Tensor};
use kernels_cuda::jit::{Ctx, Launch};
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

fn bytes_of_i32(v: &[i32]) -> Vec<u8> {
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

// ── the toy, at gemma's numbers ──────────────────────────────────────────

/// What the statement states, so a mutation can move exactly one of them.
#[derive(Clone, Copy)]
struct Stated {
    kv_heads: u32,
    head_dim: u32,
    theta: f32,
    q_eps: f32,
    k_eps: f32,
}

impl Stated {
    /// gemma-e4b's sliding layer.
    const fn gemma() -> Stated {
        Stated {
            kv_heads: 2,
            head_dim: 256,
            theta: 10_000.0,
            q_eps: 1.0e-6,
            k_eps: 1.0e-6,
        }
    }
}

/// One fire's worth of device memory: the packed rows, the two head norms,
/// the pool and its CSRs, and the two destinations.
///
/// THREE ROWS AND ONE OF THEM INVALID, and the writes are SCATTERED: the
/// three rows land on pages 2, 0 and 1 at offsets 7, 3 and 11, so a body that
/// read the write descriptors in row order out of a dense assumption would
/// put every plane in the wrong place and a body that ignored `row_valid`
/// would write a fourth.
struct Toy {
    q_heads: i32,
    rows: i32,
    pages: i32,
    page_size: i32,
    packed: Slab,
    q_weight: Slab,
    k_weight: Slab,
    positions: Slab,
    page_indices: Slab,
    page_indptr: Slab,
    last_page_lens: Slab,
    write_page: Slab,
    write_offset: Slab,
    row_valid: Slab,
    qo_indptr: Slab,
}

impl Toy {
    fn build(s: Stated, seed: u64, positions: &[i32]) -> Toy {
        let (q_heads, rows, pages, page_size) = (8i32, 3i32, 4i32, 16i32);
        let head_dim = s.head_dim as i32;
        let kv_heads = s.kv_heads as i32;
        let packed_width = (q_heads + 2 * kv_heads) * head_dim;
        let mut rng = Rng(seed);
        let packed: Vec<u16> = (0..rows * packed_width).map(|_| rng.bf16(1.5)).collect();
        let q_weight: Vec<u16> = (0..head_dim).map(|_| rng.bf16(1.0)).collect();
        let k_weight: Vec<u16> = (0..head_dim).map(|_| rng.bf16(1.0)).collect();
        Toy {
            q_heads,
            rows,
            pages,
            page_size,
            packed: Slab::of(&bytes_of_u16(&packed)),
            q_weight: Slab::of(&bytes_of_u16(&q_weight)),
            k_weight: Slab::of(&bytes_of_u16(&k_weight)),
            positions: Slab::of(&bytes_of_i32(positions)),
            // Scattered, and never `0..n`: a page index read as a row index
            // lands somewhere plausible in a dense table and nowhere here.
            page_indices: Slab::of(&bytes_of_i32(&[2, 0, 1, 3])),
            page_indptr: Slab::of(&bytes_of_i32(&[0, 2, 3, 4])),
            last_page_lens: Slab::of(&bytes_of_i32(&[8, 4, 12])),
            write_page: Slab::of(&bytes_of_i32(&[2, 0, 1])),
            write_offset: Slab::of(&bytes_of_i32(&[7, 3, 11])),
            // Row 1 is dead — a graph-padding row. The kernel's own test is
            // `row_valid != nullptr && row_valid[r] == 0`.
            row_valid: Slab::of(&[1u8, 0, 1]),
            qo_indptr: Slab::of(&bytes_of_i32(&[0, 1, 2, 3])),
        }
    }

    fn kv_dim(&self, s: Stated) -> i32 {
        s.kv_heads as i32 * s.head_dim as i32
    }

    fn page_elems(&self, s: Stated) -> usize {
        (self.pages * self.page_size * self.kv_dim(s)) as usize
    }

    fn q_elems(&self, s: Stated) -> usize {
        (self.rows * self.q_heads * s.head_dim as i32) as usize
    }

    /// The pool row this fire writes into, with the two destinations it is
    /// handed. NHD, in ELEMENTS, per `driver-cuda/src/bind/views.rs`.
    fn view(&self, s: Stated, keys: &Slab, values: &Slab) -> PagedKvView {
        PagedKvView {
            keys: keys.ptr.cast(),
            values: values.ptr.cast(),
            bf16_keys: keys.ptr.cast(),
            bf16_values: values.ptr.cast(),
            page_indices: self.page_indices.ptr.cast(),
            page_indptr: self.page_indptr.ptr.cast(),
            last_page_lens: self.last_page_lens.ptr.cast(),
            key_scales: core::ptr::null(),
            value_scales: core::ptr::null(),
            write_page: self.write_page.ptr.cast(),
            write_offset: self.write_offset.ptr.cast(),
            page_size: self.page_size,
            seq_stride: i64::from(self.kv_dim(s)),
            head_stride: i64::from(s.head_dim),
            layout: 0,
            storage_dtype: kernels_cuda::attn::KvDType::Bf16 as i32,
            scheme_byte: kernels_cuda::attn::KvScheme::Native as i32,
            native_bf16: true,
            has_envelopes: false,
            env_min: core::ptr::null(),
            env_max: core::ptr::null(),
            block_size: 0,
            max_pages_per_request: 2,
            pages_in_batch: self.pages,
            qo_indptr: self.qo_indptr.ptr.cast(),
            row_valid: self.row_valid.ptr.cast(),
            requests: self.rows,
        }
    }
}

/// What one fire leaves behind: the q rectangle and both page planes.
struct Left {
    q: Vec<u16>,
    keys: Vec<u16>,
    values: Vec<u16>,
}

// ── the two paths ────────────────────────────────────────────────────────

/// THE INHERENT METHOD, called the way the generated arm calls it — same
/// order, same marks, same columns.
fn tier2(ctx: &Ctx<'_>, toy: &Toy, s: Stated, view: &PagedKvView, q_out: &Slab) -> Result<(), Refusal> {
    let head_dim = s.head_dim as i32;
    let packed_width = (toy.q_heads + 2 * s.kv_heads as i32) * head_dim;
    ctx.qkv_fused_qknorm_rope_vnorm_write::<bf16>(
        In { ptr: toy.packed.ptr.cast(), rows: toy.rows, width: packed_width },
        In { ptr: toy.positions.ptr.cast(), rows: toy.rows, width: 1 },
        Const::new(toy.q_weight.ptr.cast::<bf16>().cast_const()),
        s.q_eps,
        Const::new(toy.k_weight.ptr.cast::<bf16>().cast_const()),
        s.k_eps,
        Cache { ptr: core::ptr::from_ref(view) },
        s.kv_heads,
        s.head_dim,
        s.theta,
        Out { ptr: q_out.ptr.cast(), rows: toy.rows, width: toy.q_heads * head_dim },
    )
}

/// THE LAUNCH THE DELETED ROUTINE BUILT, transcribed.
///
/// `attn::qkv_fused::qkv_decode_qk_norm_rope_write_kv_bf16` read its head
/// counts and its `kvc` fields exactly as below and handed them to
/// `qkv_decode_fused_dispatch`, which picked the warp instantiation off the
/// head width and ordered these twenty-two arguments. Both functions are
/// gone; this is what they said, and it is the only copy left, which is the
/// point — a second live spelling of a dispatch is the thing that drifts.
#[allow(clippy::too_many_arguments)]
fn by_name(ctx: &Ctx<'_>, toy: &Toy, s: Stated, view: &PagedKvView, q_out: &Slab) -> Result<(), Refusal> {
    const WARP_BLOCK: u32 = 256;
    const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;
    let head_dim = s.head_dim as i32;
    let kv_heads = s.kv_heads as i32;
    let packed_width = (toy.q_heads + 2 * kv_heads) * head_dim;
    let num_q_heads = (packed_width - 2 * kv_heads * head_dim) / head_dim;
    let heads = num_q_heads.unsigned_abs() + kv_heads.unsigned_abs();
    let units = toy.rows.unsigned_abs().saturating_mul(heads);
    let packed = In::<Tensor<bf16>> {
        ptr: toy.packed.ptr.cast(),
        rows: toy.rows,
        width: packed_width,
    };
    let q = Out::<Tensor<bf16>> {
        ptr: q_out.ptr.cast(),
        rows: toy.rows,
        width: num_q_heads * head_dim,
    };
    ctx.fire(
        Fire::at(
            "attn/qkv_fused.cuh",
            "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(256), false>",
        )
        .apply(Launch::grid(
            [units.div_ceil(WARPS_PER_BLOCK), 1, 1],
            [WARP_BLOCK, 1, 1],
        )),
        &[
            packed.arg(),
            q.arg(),
            view.keys.cast::<bf16>().arg(),
            view.values.cast::<bf16>().arg(),
            toy.q_weight.ptr.cast::<bf16>().cast_const().arg(),
            toy.k_weight.ptr.cast::<bf16>().cast_const().arg(),
            toy.positions.ptr.cast::<i32>().cast_const().arg(),
            core::ptr::null::<f32>().arg(),
            (view.page_indices as *const u32).arg(),
            (view.page_indptr as *const u32).arg(),
            (view.last_page_lens as *const u32).arg(),
            (view.write_page as *const u32).arg(),
            (view.write_offset as *const u32).arg(),
            view.row_valid.arg(),
            core::ptr::null::<u32>().arg(),
            toy.rows.arg(),
            num_q_heads.arg(),
            kv_heads.arg(),
            view.page_size.arg(),
            (view.layout != 0).arg(),
            s.theta.arg(),
            s.q_eps.arg(),
        ],
    )
}

/// One fire on a fresh pair of page planes, read back.
fn run(
    s: Stated,
    positions: &[i32],
    by: impl Fn(&Ctx<'_>, &Toy, Stated, &PagedKvView, &Slab) -> Result<(), Refusal>,
) -> Left {
    let _fire = FIRE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let toy = Toy::build(s, 0x9E37_79B9_7F4A_7C15, positions);
    // POISONED, NOT ZEROED: a plane the kernel never reaches keeps a value
    // no launch would produce, so "the two agree" cannot be two skips.
    let poison = vec![0xBEEFu16; toy.page_elems(s)];
    let keys = Slab::of(&bytes_of_u16(&poison));
    let values = Slab::of(&bytes_of_u16(&poison));
    let q_out = Slab::of(&bytes_of_u16(&vec![0xBEEFu16; toy.q_elems(s)]));
    let view = toy.view(s, &keys, &values);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    by(&ctx, &toy, s, &view, &q_out).expect("the fused write refused");
    Left {
        q: q_out.read_u16(toy.q_elems(s)),
        keys: keys.read_u16(toy.page_elems(s)),
        values: values.read_u16(toy.page_elems(s)),
    }
}

// ── the lock ─────────────────────────────────────────────────────────────

const POS: &[i32] = &[41, 7, 4095];

#[test]
fn the_inherent_method_builds_the_launch_the_routine_built() {
    if !device_or_skip("the tier-2 fused write") {
        return;
    }
    let s = Stated::gemma();
    let want = run(s, POS, by_name);
    let have = run(s, POS, tier2);

    // NOT VACUOUS: the fire wrote, and it wrote where the descriptors said.
    assert!(
        want.q.iter().any(|&x| x != 0xBEEF),
        "the reference launch left the q rectangle poisoned — nothing fired"
    );
    assert!(
        want.keys.iter().any(|&x| x != 0xBEEF),
        "the reference launch left the key plane poisoned — nothing was appended"
    );

    assert_eq!(have.q, want.q, "the roped q rectangle");
    assert_eq!(have.keys, want.keys, "the key plane");
    assert_eq!(have.values, want.values, "the value plane");
}

/// A DEAD ROW IS NOT WRITTEN, which is the one thing the pool row's
/// `row_valid` plane is for and the one thing a body that dropped it would
/// still pass every other assertion without.
#[test]
fn the_invalid_row_leaves_its_page_untouched() {
    if !device_or_skip("the tier-2 fused write") {
        return;
    }
    let s = Stated::gemma();
    let left = run(s, POS, tier2);
    // Row 1 is invalid and its descriptors name page 0, offset 3.
    let kv_dim = (s.kv_heads * s.head_dim) as usize;
    let at = (0 * 16 + 3) * kv_dim;
    assert!(
        left.keys[at..at + kv_dim].iter().all(|&x| x == 0xBEEF),
        "the invalid row's key page was written"
    );
    // And its q heads ARE written: the early-out is on the kv side only,
    // which is what the kernel says (`if (!is_q && row_valid ...)`).
    let q_row = (1 * 8 * s.head_dim) as usize;
    assert!(
        left.q[q_row..q_row + s.head_dim as usize]
            .iter()
            .any(|&x| x != 0xBEEF),
        "the invalid row's q head was skipped, and only its kv write should be"
    );
}

/// EVERY SCALAR THE STATEMENT CARRIES MOVES THE ANSWER.
///
/// One mutation per column the generated arm reads by index. A column read
/// one place off — `q_eps` where `k_eps` stands, `head_dim` where `kv_heads`
/// does — is a wrong answer that no shape check sees, and a mutation that did
/// not move the bytes is how it would hide.
#[test]
fn every_stated_scalar_reaches_the_kernel() {
    if !device_or_skip("the tier-2 fused write") {
        return;
    }
    let base = Stated::gemma();
    let want = run(base, POS, tier2);

    let eps = Stated { q_eps: 0.5, k_eps: 0.5, ..base };
    assert_ne!(run(eps, POS, tier2).q, want.q, "the head-norm epsilon");

    let theta = Stated { theta: 1_000_000.0, ..base };
    assert_ne!(run(theta, POS, tier2).q, want.q, "the rope base");

    // The head counts re-cut the packed row, so the q plane moves.
    let heads = Stated { kv_heads: 4, ..base };
    assert_ne!(run(heads, POS, tier2).q, want.q, "the kv head count");

    // The position is a per-ROW operand, not a scalar, and it is the last
    // column the arm reads off `inputs`.
    let moved: &[i32] = &[40, 7, 4095];
    assert_ne!(run(base, moved, tier2).q, want.q, "the per-row position");
}

/// THE TWO EPSILONS, REFUSED BY NAME when they disagree.
///
/// The statement states two because it names two `Norm`s; the kernel takes
/// one. Serving the pair by dropping one would normalise k at q's epsilon,
/// which is a plausible wrong answer — `attention.masked`'s
/// window-beside-a-mask, exactly.
#[test]
fn two_epsilons_that_disagree_are_refused() {
    if !device_or_skip("the tier-2 fused write") {
        return;
    }
    let s = Stated { k_eps: 1.0e-5, ..Stated::gemma() };
    let toy = Toy::build(s, 1, POS);
    let keys = Slab::of(&bytes_of_u16(&vec![0u16; toy.page_elems(s)]));
    let values = Slab::of(&bytes_of_u16(&vec![0u16; toy.page_elems(s)]));
    let q_out = Slab::of(&bytes_of_u16(&vec![0u16; toy.q_elems(s)]));
    let view = toy.view(s, &keys, &values);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let refusal = tier2(&ctx, &toy, s, &view, &q_out).expect_err("two epsilons were served");
    assert!(
        matches!(refusal, Refusal::Unstated { what } if what.contains("epsilon")),
        "refused, but not by naming the pair: {refusal:?}"
    );
}

/// A POOL ROW THIS STATEMENT NAMES AND THIS FIRE DID NOT STAGE.
#[test]
fn a_null_pool_row_refuses_rather_than_faulting() {
    if !device_or_skip("the tier-2 fused write") {
        return;
    }
    let s = Stated::gemma();
    let toy = Toy::build(s, 2, POS);
    let q_out = Slab::of(&bytes_of_u16(&vec![0u16; toy.q_elems(s)]));
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let head_dim = s.head_dim as i32;
    let packed_width = (toy.q_heads + 2 * s.kv_heads as i32) * head_dim;
    let refusal = ctx
        .qkv_fused_qknorm_rope_vnorm_write::<bf16>(
            In { ptr: toy.packed.ptr.cast(), rows: toy.rows, width: packed_width },
            In { ptr: toy.positions.ptr.cast(), rows: toy.rows, width: 1 },
            Const::new(toy.q_weight.ptr.cast::<bf16>().cast_const()),
            s.q_eps,
            Const::new(toy.k_weight.ptr.cast::<bf16>().cast_const()),
            s.k_eps,
            Cache::<Struct<KvCache>> { ptr: core::ptr::null() },
            s.kv_heads,
            s.head_dim,
            s.theta,
            Out { ptr: q_out.ptr.cast(), rows: toy.rows, width: toy.q_heads * head_dim },
        )
        .expect_err("a null pool row was served");
    assert!(
        matches!(refusal, Refusal::Null { .. }),
        "a null pool row refused with {refusal:?}"
    );
}

/// THE ELEMENT PIN. The generated arm instantiates the mechanical `Elem^1`
/// cartesian — bf16 AND f32 — and the f32 arm is a refusal by name, because
/// the kernel behind this is spelled at bf16 and nowhere else.
#[test]
fn an_element_this_kernel_is_not_written_for_refuses() {
    if !device_or_skip("the tier-2 fused write") {
        return;
    }
    let s = Stated::gemma();
    let toy = Toy::build(s, 3, POS);
    let keys = Slab::of(&bytes_of_u16(&vec![0u16; toy.page_elems(s)]));
    let values = Slab::of(&bytes_of_u16(&vec![0u16; toy.page_elems(s)]));
    let q_out = Slab::of(&bytes_of_u16(&vec![0u16; toy.q_elems(s) * 2]));
    let view = toy.view(s, &keys, &values);
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let head_dim = s.head_dim as i32;
    let packed_width = (toy.q_heads + 2 * s.kv_heads as i32) * head_dim;
    let refusal = ctx
        .qkv_fused_qknorm_rope_vnorm_write::<f32>(
            In { ptr: toy.packed.ptr.cast(), rows: toy.rows, width: packed_width },
            In { ptr: toy.positions.ptr.cast(), rows: toy.rows, width: 1 },
            Const::new(toy.q_weight.ptr.cast::<f32>().cast_const()),
            s.q_eps,
            Const::new(toy.k_weight.ptr.cast::<f32>().cast_const()),
            s.k_eps,
            Cache { ptr: core::ptr::from_ref(&view) },
            s.kv_heads,
            s.head_dim,
            s.theta,
            Out { ptr: q_out.ptr.cast(), rows: toy.rows, width: toy.q_heads * head_dim },
        )
        .expect_err("f32 was served");
    assert!(
        matches!(refusal, Refusal::Absent { what } if what.contains("element other than bf16")),
        "refused, but not at the element: {refusal:?}"
    );
}

/// THE BLOCK ARM, at a width the warp arm has no register tiling for.
///
/// The method picks between two instantiations on the head width alone — 64,
/// 128 and 256 take the warp form, everything else the 128-wide block form
/// that reads the width as a runtime argument. gemma takes the first; this is
/// the second, so both are reachable and neither is a guess.
#[test]
fn the_block_arm_answers_a_width_the_warp_arm_has_no_tiling_for() {
    if !device_or_skip("the tier-2 fused write") {
        return;
    }
    let s = Stated { head_dim: 192, ..Stated::gemma() };
    let left = run(s, POS, tier2);
    assert!(
        left.q.iter().any(|&x| x != 0xBEEF),
        "the block arm left the q rectangle poisoned — nothing fired"
    );
    assert!(
        left.keys.iter().any(|&x| x != 0xBEEF),
        "the block arm appended nothing"
    );
}
