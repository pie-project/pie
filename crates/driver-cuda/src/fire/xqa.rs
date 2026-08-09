//! XQA's two fires, and the workspace carve that is the only thing between
//! them that the DRIVER owns.
//!
//! # Every `attention_xqa*.cu:NNN` citation below is a line in a DELETED file
//!
//! Said once here rather than forty times inline, which is the same device
//! [`super::mla_naive`] uses. The six archive translation units this file was
//! the host half of —
//!
//! ```text
//!   csrc/src/attn/attention_xqa.cu           436 lines
//!   csrc/src/attn/attention_xqa_gqa2.cu      194
//!   csrc/src/attn/attention_xqa_gqa2_p16.cu  194
//!   csrc/src/attn/attention_xqa_gqa4.cu      194
//!   csrc/src/attn/attention_xqa_gqa8.cu      244
//!   csrc/src/attn/attention_xqa_gqa8_sm90.cu 197
//! ```
//!
//! — are **deleted**, along with `csrc/src/attn/attention_flashinfer_hopper_stub.cpp`,
//! which was chained to them. Their line numbers were correct against the last
//! commit that contained them, `0dc8e9e9b`, and `git show
//! 0dc8e9e9b:crates/kernels-cuda/csrc/src/attn/attention_xqa.cu` is how a
//! reader checks one.
//!
//! The citations are kept rather than stripped because each one is evidence
//! for a decision, and a decision whose evidence has been deleted is a
//! decision no one can re-check. They are **quotations from history**, not
//! pointers into the tree; nothing automated should try to resolve them.
//!
//! # What is left here, and why it is only this
//!
//! Everything that picks a symbol or computes a grid is
//! [`kernels_cuda_new::x::xqa`]: the five roots, the lattice member a shape
//! selects, the SM90 gate, the page bucket, the multi-block split, the strides
//! in heads, the 79,488-byte shared-memory ask, and the semaphore memset that
//! has to be on the stream ahead of the launch. Two routines, one per fire.
//!
//! What stays is what reads the DRIVER's vocabulary, and that is one thing:
//! [`AttentionWorkspaceView`]. XQA's page table, its sequence lengths and its
//! own scratch are three sub-buffers of the caller's `float_buffer`, laid out
//! end to end by [`carve`] — and `Source` names buffers, not offsets into
//! them, so the carve is arithmetic over a base address that no query and no
//! rule can see.
//!
//! **The carve is the coupling, and it is the one thing here that can be
//! silently wrong.** [`prepare_decode`]'s kernel WRITES the page table and the
//! sequence lengths; [`decode`] READS them back. One [`carve`] serves both, so
//! the two cannot disagree — where the archive had five copies of this
//! arithmetic in four files, joined by nothing a compiler checks.
//!
//! # Ordering
//!
//! [`prepare_decode`] and [`decode`] must be on ONE stream, in that order.
//! Nothing states that dependency — two symbols state two geometries and no
//! edge between them — so it is the caller's, exactly as it was when both
//! halves were C++ on the same `cudaStream_t`.
//!
//! # The obligation, and why deleting the C++ was the point
//!
//! `attn::attention_xqa_decode_bf16_prepared` states `needs =
//! Prepare::FireWide`. A `Prepare` is an obligation written in the TABLE and
//! discharged by the driver — there is no call edge for a reachability audit
//! to find, which is why `new-horizon.md` §44.5 recorded the C++
//! implementation as unreachable-and-kept and called that the audit's third
//! blind spot.
//!
//! **Nothing calls either function yet, and neither did the C++.**
//! `Prepare::FireWide` is read by no code in this repository: not
//! `model-compiler`, not `bind::dispatch`, not `fire::launch`. The obligation
//! has been undischarged on every channel since the row was written; what
//! changed is that the implementation is Rust, fires NVRTC-compiled text, and
//! is reachable from a `#[cfg(feature = "_cuda")]` build with no archive
//! linked. Wiring the call is a `fire::launch` change and belongs to whoever
//! routes `FireWide`; `bind/arms/xqa.rs` carries the same sentence at the
//! symbol.
//!
//! # THE JIT'D XQA DOES NOT MATCH THE ARCHIVE BIT FOR BIT
//!
//! `runtime::nvrtc::options` passes `--fmad=false --prec-div=true
//! --prec-sqrt=true` on every compile and the archive's CMake passed none of
//! the three, so nvcc built these same instantiations under `--fmad=true`: it
//! contracts multiply-adds and the JIT refuses to. The JIT is the stricter of
//! the two and the direction of the disagreement is known, but a
//! bit-exactness claim would be false by construction (`new-horizon.md`
//! §62.8).

use kernels::Refusal;
use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::xqa;

use crate::bind::abi::AttentionWorkspaceView;

/// The scratch's alignment — `attention_xqa.cu:49`'s
/// `constexpr std::size_t kSemaphoreAlignment = 256`.
///
/// It aligns the tail of the carve, which is the region the decode hands XQA
/// as its scratch. Named for the semaphores because the same 256 governs the
/// semaphore bank the decode zeroes out of `int_buffer`; only the alignment is
/// shared, not the buffer.
const SCRATCH_ALIGN: usize = 256;

/// Why [`prepare_decode`] did not launch.
///
/// One arm, and it stays an enum rather than collapsing to a `bool` for the
/// reason [`super::gemv::Decline`] gives: a caller must not be able to spell
/// *"it declined"* the same way it spells *"it ran"*, and the shape of the
/// answer should not change the day a second refusal is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    /// `num_requests <= 0` — `attention_xqa.cu:290`'s `if (num_requests <= 0) return;`.
    ///
    /// An empty fire is not an error: there is no request to build a page
    /// table for and the decode that follows declines for the same reason.
    NoRequests,
}

/// What [`prepare_decode`] did.
///
/// `#[must_use]` because ignoring this answer is the one way to get a wrong
/// result out of that function: a declined call enqueues nothing at all, and a
/// decode fired after one reads a page table that was never written — which is
/// not a fault, it is whatever the workspace held from the last fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum XqaPrepare {
    /// The build is on the stream. The page table and the sequence lengths are
    /// written by the time anything else ordered on that stream reads them.
    Prepared,
    /// Nothing was enqueued.
    Declined(Decline),
}

/// Where the three XQA sub-buffers sit inside `float_buffer`.
///
/// Every field is a device address. Held together in one value because the
/// three are one layout: computing any of them without the others is how the
/// archive's five copies of this arithmetic got to disagree.
#[derive(Debug, Clone, Copy)]
pub struct Carve {
    /// `num_requests * page_bucket` signed page indices, zero padded.
    pub page_table: *mut i32,
    /// One `u32` per request.
    pub seq_lens: *mut u32,
    /// Everything after them, 256-byte aligned — XQA's own scratch.
    pub scratch: *mut std::ffi::c_void,
    /// The row stride [`kernels_cuda_new::x::xqa::page_bucket`] chose, which
    /// the decode needs twice: as the table's stride and, times `page_size`,
    /// as XQA's maximum sequence length.
    ///
    /// Carried rather than recomputed at the fire so that the number the
    /// carve sized for and the number the kernel strides by are one value.
    pub page_bucket: i32,
}

/// The workspace carve — `attention_xqa.cu:291-309`, transcribed.
///
/// ```text
/// page_table_bytes = num_requests * page_bucket * sizeof(i32)
/// seq_lens_bytes   = num_requests * sizeof(u32)
/// p_page_table = align_up(float_buffer,                    alignof(i32))
/// p_seq_lens   = align_up(p_page_table + page_table_bytes, alignof(u32))
/// p_scratch    = align_up(p_seq_lens   + seq_lens_bytes,   256)
/// if (p_scratch >= float_buffer + float_bytes) throw
/// ```
///
/// # The refusal is `>=`, not `>`
///
/// Carried exactly. A scratch that starts at the last byte of the workspace
/// has nothing in it, so the C++ refused an EMPTY scratch as well as an
/// overflowing one, and it refused it before any launch. Relaxing this to `>`
/// would hand XQA a zero-length region that it would write through.
///
/// # Panics
///
/// If the three regions do not fit in `float_bytes`, naming what was needed
/// and what there was. `attention_xqa.cu:308` threw `"xqa decode: attention
/// workspace too small"`; the shim that caught it is gone, so this is a panic
/// rather than a decline — the same choice [`super::attn_score`] makes for the
/// guard the C++ threw on, and for the same reason. A prepare that quietly did
/// not run leaves the decode reading the previous fire's page table, which is
/// a plausible answer rather than a missing one.
///
/// Also panics on a null `float_buffer`. **The C++ made no such test**, and
/// the arithmetic above only caught it when `float_bytes` was zero; a
/// workspace that is null and claims bytes would have produced a launch
/// against address 0, reported asynchronously as an unrelated fault at
/// whatever synchronised next.
#[must_use]
pub fn carve(workspace: AttentionWorkspaceView, num_requests: i32, max_pages: i32) -> Carve {
    assert!(
        !workspace.float_buffer.is_null(),
        "xqa prepare: the attention workspace has no float buffer (float_bytes={})",
        workspace.float_bytes
    );
    let bucket = xqa::page_bucket(max_pages);
    let requests = num_requests.unsigned_abs() as usize;
    let page_table_bytes = requests * bucket.unsigned_abs() as usize * size_of::<i32>();
    let seq_lens_bytes = requests * size_of::<u32>();

    let base = workspace.float_buffer.addr();
    let p_page_table = align_up(base, align_of::<i32>());
    let p_seq_lens = align_up(p_page_table + page_table_bytes, align_of::<u32>());
    let p_scratch = align_up(p_seq_lens + seq_lens_bytes, SCRATCH_ALIGN);
    let end = base + workspace.float_bytes;
    assert!(
        p_scratch < end,
        "xqa prepare: attention workspace too small — the carve needs more than {} bytes \
         for {num_requests} requests at a {bucket}-page stride (page table {page_table_bytes} B, \
         sequence lengths {seq_lens_bytes} B, then a {SCRATCH_ALIGN}-byte-aligned scratch)",
        workspace.float_bytes
    );

    Carve {
        page_table: workspace.float_buffer.with_addr(p_page_table).cast(),
        seq_lens: workspace.float_buffer.with_addr(p_seq_lens).cast(),
        scratch: workspace.float_buffer.with_addr(p_scratch),
        page_bucket: bucket,
    }
}

/// `attention_xqa.cu:51-53` — `align_up_ptr(p, a)`, whose body is
/// `(p + a - 1) / a * a`.
///
/// Spelled with [`usize::next_multiple_of`] rather than transcribed, because
/// the two agree for every `a > 0` and the standard-library name is the one a
/// reader does not have to check. A host computation over an ADDRESS, which is
/// why it is here at all: no [`kernels::Source`] and no [`kernels::LaunchRule`]
/// can see one.
fn align_up(p: usize, a: usize) -> usize {
    p.next_multiple_of(a)
}

/// Build XQA's dense page table and sequence lengths into the fire's
/// attention workspace.
///
/// The carve, and then one call to
/// [`kernels_cuda_new::x::xqa::build_xqa_metadata`]. Everything the deleted
/// `prepare_attention_xqa_decode_bf16` did between those two is in the
/// routine: the bucket, the block width, the grid.
///
/// # Panics
///
/// If the carve does not fit (see [`carve`]), or if the build refuses — which
/// after the empty-fire test above can only be the compile, the load or the
/// launch. None of those is a shape this function may decline over: there is
/// no ahead-of-time launcher left to fall back to and no second answer to
/// give.
///
/// # What the caller is still asserting
///
/// Not `unsafe`, and no pointer is dereferenced here — they are compared
/// against null and handed to the launch as addresses. The caller asserts,
/// exactly as it did when it handed the same three pointers and a
/// `cudaStream_t` to a C++ launcher, that they are device addresses of the
/// stated extents and that `stream` is live across the launch.
#[allow(clippy::too_many_arguments)] // the C++ launcher's parameter list, unchanged
#[allow(clippy::not_unsafe_ptr_arg_deref)] // nothing here dereferences one
pub fn prepare_decode(
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    num_requests: i32,
    page_size: i32,
    max_pages_per_seq: i32,
    workspace: AttentionWorkspaceView,
    stream: *mut std::ffi::c_void,
) -> XqaPrepare {
    // `attention_xqa.cu:290` — `if (num_requests <= 0) return;`. Caught HERE
    // as well as in the routine, because this function's answer is an enum a
    // caller matches on and "it declined because there was nothing to do" is
    // not the same fact as "the launch refused".
    if num_requests <= 0 {
        return XqaPrepare::Declined(Decline::NoRequests);
    }
    assert!(
        !kv_page_indices_d.is_null()
            && !kv_page_indptr_d.is_null()
            && !kv_last_page_lens_d.is_null(),
        "xqa prepare: the page CSR must be three device pointers \
         (indices={kv_page_indices_d:?}, indptr={kv_page_indptr_d:?}, \
         last_page_lens={kv_last_page_lens_d:?})"
    );

    let regions = carve(workspace, num_requests, max_pages_per_seq);

    // SAFETY: the caller holds `stream` live across the launch — the same
    // assertion it made when it handed `stream` to a C++ launcher that put it
    // in a `<<<>>>`.
    let ctx = unsafe { Ctx::on(stream) };
    if let Err(why) = xqa::build_xqa_metadata(
        &ctx,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        regions.page_table,
        regions.seq_lens,
        num_requests,
        max_pages_per_seq,
        page_size,
    ) {
        panic!("xqa prepare: the metadata build refused: {why}");
    }
    XqaPrepare::Prepared
}

/// Fire XQA's decode against the page table [`prepare_decode`] wrote.
///
/// The carve read back, the semaphore budget checked against the workspace's
/// int half, and then one call to
/// [`kernels_cuda_new::x::xqa::xqa_decode_bf16`]. The shape gate, the lattice
/// member, the grid, the strides and the semaphore memset are all in the
/// routine; what is here is the two things the routine cannot see, which are
/// both the workspace.
///
/// `sm_scale` is CHECKED and never applied — XQA folds `1/sqrt(head_dim)` into
/// the kernel — so a caller that means a different scale is refused rather
/// than served. See the routine's `decode_supported`.
///
/// # Errors
///
/// [`Refusal::Wide`] if the int half cannot hold `num_requests *
/// num_kv_heads` semaphores (`attention_xqa_gqa2.cu:150` threw `"xqa gqa2
/// decode: semaphore workspace too small"`), and whatever the routine
/// refuses: an unserved shape or device, a lattice member with no root, an
/// empty request set, or a compile, load or launch that would not go.
///
/// # Panics
///
/// If the carve does not fit — see [`carve`], which is where the C++ threw and
/// where the panic is argued. The old planner tested `scratch >= end` a second
/// time here and returned a decline for it; that branch was dead, because
/// `carve` asserts the same condition before it can be reached.
///
/// # What the caller is still asserting
///
/// [`prepare_decode`]'s, plus: `q` and `o` address `num_requests *
/// num_q_heads` heads each, `k_pages` and `v_pages` the layer's page arena,
/// and this call is ordered on the same stream AFTER the prepare.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::not_unsafe_ptr_arg_deref)] // nothing here dereferences one
pub fn decode(
    q: *const std::ffi::c_void,
    o: *mut std::ffi::c_void,
    k_pages: *mut std::ffi::c_void,
    v_pages: *mut std::ffi::c_void,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    max_pages_per_seq: i32,
    sm_scale: f32,
    workspace: AttentionWorkspaceView,
    stream: *mut std::ffi::c_void,
) -> Result<(), Refusal> {
    if num_requests <= 0 {
        return Err(Refusal::Empty { what: "num_requests" });
    }
    // The same carve the prepare wrote, read back rather than recomputed — the
    // page table the decode reads has to be the one the prepare filled, and
    // two copies of this arithmetic is exactly what the archive had.
    let regions = carve(workspace, num_requests, max_pages_per_seq);

    // `attention_xqa_gqa2.cu:145-151` — the bank lives in the workspace's int
    // half, so its budget is a workspace question and stays on this side. The
    // routine zeroes it; only the caller knows how much room there is.
    let semaphores = num_requests.unsigned_abs() as usize * num_kv_heads.unsigned_abs() as usize;
    let semaphore_bytes = semaphores * size_of::<u32>();
    if semaphore_bytes > workspace.int_bytes {
        return Err(Refusal::Wide {
            what: "the XQA semaphore bank",
            at: i64::try_from(semaphore_bytes).unwrap_or(i64::MAX),
            max: i64::try_from(workspace.int_bytes).unwrap_or(i64::MAX),
        });
    }

    // SAFETY: the caller holds `stream` live across the launch, and every
    // pointer below is a device address of the extent this function's contract
    // states.
    let ctx = unsafe { Ctx::on(stream) };
    xqa::xqa_decode_bf16(
        &ctx,
        q.cast(),
        o.cast(),
        k_pages,
        v_pages,
        regions.page_table.cast_const(),
        regions.seq_lens.cast_const(),
        workspace.int_buffer.cast(),
        regions.scratch,
        num_requests,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_pages_per_seq,
        sm_scale,
    )
}
