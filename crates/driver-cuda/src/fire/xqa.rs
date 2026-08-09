//! XQA's fire-wide prepare: the host half of `attn/attention_xqa.cu`, in Rust.
//!
//! # The specification of this program, in four terms
//!
//! Written against the code below so a reader can check the code against a
//! claim rather than infer the claim from the code — the same shape
//! [`super::gemv`]'s header takes, and for the same reason.
//!
//! **1. Which JIT units it fires, in what order.** One unit,
//! `attn/attention_xqa`, one row, one fire. There is no order and no
//! alternative: [`prepare_decode`] either enqueues `attn::build_xqa_metadata`
//! once or enqueues nothing. This is not a composition and needs no
//! `execution::Step`.
//!
//! **2. What intermediate buffers sit between them.** None between kernels,
//! and three carved BEFORE the one — which is the whole reason this function
//! exists. `page_table`, `seq_lens` and a 256-byte-aligned `scratch` are
//! sub-buffers of the caller's `AttentionWorkspaceView::float_buffer`, laid
//! out end to end by [`carve`]. This function allocates nothing and frees
//! nothing; the workspace is the caller's and the carve is arithmetic over its
//! base address.
//!
//! **The carve is the coupling, and it is the one thing here that can be
//! silently wrong.** The kernel WRITES the page table and the sequence
//! lengths; `attn::attention_xqa_decode_bf16_prepared` — still ahead-of-time,
//! still the shim's one XQA entry — READS them back, and it recomputes the
//! same offsets itself rather than being handed them
//! (`attention_xqa.cu:420-442`, and once more in each of the four
//! `detail::launch_attention_xqa_decode_bf16_gqa*_prepared` bodies). Five
//! copies of one layout, in four files, joined by nothing a compiler checks.
//! [`carve`] is the sixth and is `pub` so it can become the first: a port of
//! the decode side takes this function rather than writing a seventh.
//!
//! **3. What it decides on the host.** Two decisions and one refusal:
//!
//! ```text
//!   #  what                                     from              picks
//!   1  num_requests <= 0                        operand           DECLINE
//!   2  page_bucket = next pow2 >= pages, <=4096 operand, const    row stride
//!   3  the carve does not fit float_bytes       operands          PANIC
//! ```
//!
//! Only 2 picks anything. 1 is the empty fire, which is legal and must not
//! reach a launch — a zero `grid.x` is `Error::Geometry` at
//! [`kernels_cuda_new::runtime::KernelModule::fire`] and would turn a legal
//! no-op into a refusal. 3 is a panic because the C++ threw
//! (`attention_xqa.cu:308`), and this crate's C++ threw through a shim that
//! caught; the catch is gone with the shim, so the refusal is spelled here or
//! it is not spelled at all.
//!
//! **4. What in `Source` / `LaunchRule` / `Specialisation` / `Execution` is
//! missing to state it.** In one line each:
//!
//! * **`LaunchRule`** — one axis short. [`kernels::LaunchRule::PerRequest`]
//!   opens `[requests, 1, 1]`, which is this grid exactly, at a block of 256;
//!   the launcher is 128 wide. `families::attn`'s `ATTN_XQA_SIGS` carries the
//!   refusal at length: the width is a pure stride, so 256 computes the same
//!   bytes, and a rule chosen by measuring bytes would be chosen by a
//!   measurement that cannot fail. A `PerRequest(block)` would be the second
//!   parameterised rule this family has been asked for in a week — see
//!   `super::attn_score`'s `FOLD_GRID_Y` for the first — and `new-horizon.md`
//!   §10.5 refuses one for one.
//! * **`Source`** — no binding for two of eight operands. `page_table` and
//!   `seq_lens` are offsets into a workspace, and `Source` names buffers, not
//!   offsets into them; `max_pages_per_seq` as the kernel reads it is the
//!   BUCKETED stride, not the caller's number, and `Source` has `Mul` and
//!   `Div` but no next-power-of-two. So the row is left unsourced whole, which
//!   is `families/rope.rs`' rule rather than an omission.
//! * **`Specialisation`** — nothing missing and nothing wanted. One
//!   instantiation, no arms, no `Term`.
//! * **`Execution`** — this is not a `Walk` and must not be made one. A
//!   `Walk`'s shape comes from the input; this program's shape is fixed. What
//!   it actually is has no arm at all, because it is not a symbol a model text
//!   states: it is the discharge of a `Prepare` that a DIFFERENT symbol's row
//!   declares. See *The obligation* below.
//!
//! # The obligation, and why deleting the C++ was the point
//!
//! `kernels_cuda_new::table::attn`'s `attn::attention_xqa_decode_bf16_prepared`
//! states `needs = Prepare::FireWide`. A `Prepare` is an obligation written in
//! the TABLE and discharged by the driver — there is no call edge for a
//! reachability audit to find, which is why `new-horizon.md` §44.5 recorded
//! the C++ implementation as unreachable-and-kept and called that the audit's
//! third blind spot.
//!
//! It was kept because it was the only text in the tree that wrote the page
//! table and the sequence lengths at the offsets the prepared launcher reads
//! back. **This module is that text now**, so the keeper is discharged and the
//! C++ is deleted — with its `__global__`, which was the last one the
//! `kernels-cuda` archive held.
//!
//! **Nothing calls this yet, and neither did the C++.** `Prepare::FireWide` is
//! read by no code in this repository: not `model-compiler`, not
//! `bind::dispatch`, not `fire::launch`. The obligation has been undischarged
//! on every channel since the row was written; what changed is that the
//! implementation is now Rust, fires NVRTC-compiled text, and is reachable
//! from a `#[cfg(feature = "_cuda")]` build with no archive linked. Wiring the
//! call is a `fire::launch` change and belongs to whoever routes `FireWide`.
//!
//! # The measurements this port carries out of a condemned file
//!
//! Two, both from `attention_xqa.cu`, recorded here because the file they are
//! in is going and a measurement that dies with its file was never a
//! measurement.
//!
//! * **The page BUCKET is a power of two on purpose** (`:274-279`). The row
//!   stride of the dense page table has to be stable across the small per-step
//!   changes in `max_pages_per_seq` that a growing decode produces, because
//!   the decode hands XQA `page_bucket * page_size` as the maximum sequence
//!   length and re-shaping the buffer every fire would invalidate a captured
//!   graph's baked addresses. It is clamped at 4096.
//! * **XQA is refused below SM90 on this deployment, and that is a run rather
//!   than a rule** (`:268-272`). FlashInfer's own public XQA wrapper only
//!   enables the path on SM90+; the Ampere/Ada csrc instantiations *compile*,
//!   and local SM89 TP2 serving runs were observed to **spin indefinitely
//!   after graph capture**, so those devices stay on the regular decode path.
//!   That predicate is `xqa_decode_bf16_supported` and is still C++, because
//!   the decode launcher that calls it is still C++; this note is here so the
//!   observation survives the file.
//!
//! # What this does NOT move
//!
//! `attention_xqa.cu` keeps `xqa_decode_bf16_supported`,
//! `xqa_decode_page_bucket` and `attention_xqa_decode_bf16_prepared`, and its
//! five sibling translation units keep theirs. Every one of them ends in
//! `launchMHAFlashInfer_xqa_gqa*_bf16_p*_h128` — an upstream FlashInfer HOST
//! function that does its own launching, reached by `#include <xqa/mha.cu>`
//! into a translation unit that renames it. `new-horizon.md` §50.1's
//! measurement applies to those unchanged: there is no device text of ours in
//! them, so the §48 split is degenerate and each becomes Rust in its entirety
//! or it does not move at all. §50.9's gap is what remains.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::bind::abi::AttentionWorkspaceView;

/// The metadata build's symbol, in the JIT table.
///
/// Resolved through `unit_of` rather than declared as a path here, so a rename
/// in `kernels-cuda-new` is a refusal at this call and not a silent miss.
const METADATA_SYMBOL: &str = "attn::build_xqa_metadata";

/// The metadata build's block width: `attention_xqa.cu:313`'s literal `128`.
///
/// **This constant is the reason the prepare is fired by hand.** It is not an
/// extent and it is not a reduction width — the page loop is
/// `for (p = threadIdx.x; p < max_pages_per_seq; p += blockDim.x)` and the
/// sequence length is written under `if (threadIdx.x == 0)`, so every width
/// computes the same page table and the same sequence lengths. That is exactly
/// what makes it unsafe to hand to a rule: `LaunchRule::PerRequest` states the
/// same `[requests, 1, 1]` grid at 256 threads and would be byte-identical and
/// wrong only in occupancy, so no parity measurement could ever refuse it.
/// `families::attn`'s `ATTN_XQA_SIGS` carries the argument; the short form is
/// that the number is a citation, not a derivation, and it sits one `git grep`
/// from the `<<<>>>` it was copied from.
const METADATA_BLOCK: u32 = 128;

/// The scratch's alignment — `attention_xqa.cu:49`'s
/// `constexpr std::size_t kSemaphoreAlignment = 256`.
///
/// It aligns the tail of the carve, which is the region the decode hands XQA
/// as its scratch. Named for the semaphores because the same 256 governs the
/// semaphore bank the decode memsets out of `int_buffer`; only the alignment
/// is shared, not the buffer.
const SCRATCH_ALIGN: usize = 256;

/// The largest page-table row stride the bucket will grow to —
/// `attention_xqa.cu:277`'s `bucket < 4096`.
const MAX_PAGE_BUCKET: i32 = 4096;

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
    /// table for and the decode that follows will decline for the same reason.
    /// It is caught HERE because a zero `grid.x` reaching
    /// [`kernels_cuda_new::runtime::KernelModule::fire`] is `Error::Geometry`,
    /// which would turn a legal no-op into a refusal.
    NoRequests,
}

/// What [`prepare_decode`] did.
///
/// `#[must_use]` because ignoring this answer is the one way to get a wrong
/// result out of this function: a declined call enqueues nothing at all, and a
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

/// The dense page table's row stride — `attention_xqa.cu:274-279`, verbatim.
///
/// ```text
/// int bucket = 1;
/// const int pages = std::max(1, max_pages_per_seq);
/// while (bucket < pages && bucket < 4096) bucket <<= 1;
/// return bucket;
/// ```
///
/// The clamp is on `bucket`, not on `pages`, so a request set wanting more
/// than 4096 pages gets 4096 and the decode's own shape checks are what refuse
/// it. Transcribed as a loop rather than as `next_power_of_two` because the
/// two differ at the clamp: `4097usize.next_power_of_two()` is 8192, and this
/// answers 4096.
#[must_use]
pub fn page_bucket(max_pages_per_seq: i32) -> i32 {
    let mut bucket = 1i32;
    let pages = max_pages_per_seq.max(1);
    while bucket < pages && bucket < MAX_PAGE_BUCKET {
        bucket <<= 1;
    }
    bucket
}

/// Where the three XQA sub-buffers sit inside `float_buffer`.
///
/// Every field is a device address. Held together in one value because the
/// three are one layout: computing any of them without the others is how the
/// five existing copies of this arithmetic get to disagree.
#[derive(Debug, Clone, Copy)]
pub struct Carve {
    /// `num_requests * page_bucket` signed page indices, zero padded.
    pub page_table: *mut i32,
    /// One `u32` per request.
    pub seq_lens: *mut u32,
    /// Everything after them, 256-byte aligned — XQA's own scratch.
    pub scratch: *mut std::ffi::c_void,
    /// The row stride [`page_bucket`] chose, which the decode needs twice: as
    /// the table's stride and, times `page_size`, as XQA's maximum sequence
    /// length.
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
/// and what there was. `attention_xqa.cu:308` threw
/// `"xqa decode: attention workspace too small"`; the shim that caught it is
/// gone, so this is a panic rather than a `Decline` — the same choice
/// [`super::attn_score`] makes for the guard the C++ threw on, and for the
/// same reason. A prepare that quietly did not run leaves the decode reading
/// the previous fire's page table, which is a plausible answer rather than a
/// missing one.
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
    let bucket = page_bucket(max_pages);
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
/// why it is here at all: no [`kernels::Source`] and no
/// [`kernels::LaunchRule`] can see one.
fn align_up(p: usize, a: usize) -> usize {
    p.next_multiple_of(a)
}

/// Build XQA's dense page table and sequence lengths into the fire's
/// attention workspace.
///
/// Ports `kernels::attn::prepare_attention_xqa_decode_bf16`
/// (`attention_xqa.cu:280-323`) — one host launcher over one `<<<>>>`, plus
/// the carve above it. The launcher, its declaration in `attn/attention_xqa.hpp`
/// and its `__global__` are DELETED; the `__global__` is
/// `kernels-cuda-new`'s `attn/attention_xqa` unit, which NVRTC compiles, and
/// everything else is here.
///
/// # What it is FOR, since nothing in the signature says
///
/// XQA's decode entry point wants a dense zero-padded page table and a flat
/// `seq_lens` array; the paged KV cache carries a ragged CSR. This is the
/// transform, and it runs once per FIRE rather than once per layer — which is
/// what `Prepare::FireWide` names and why
/// `attn::attention_xqa_decode_bf16_prepared` is also `whole`.
///
/// # Ordering
///
/// The build and the decode that reads it back must be on ONE stream, in that
/// order. Nothing states that dependency — two symbols state two geometries
/// and no edge between them, which is `PAGE_COMPACT_ROWS`' situation next door
/// — so it is the caller's, exactly as it was when both halves were C++ on the
/// same `cudaStream_t`.
///
/// # Panics
///
/// If the carve does not fit (see [`carve`]), if `attn/attention_xqa` is in no
/// JIT unit, if the unit will not compile or load, or if the row's operands
/// and the kernel's parameters have drifted. None of those is a shape this
/// function may decline over: `emit_c_shim` emits no entry for this row, so
/// there is no ahead-of-time launcher left to fall back to and no second
/// answer to give.
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
    // `attention_xqa.cu:290` — `if (num_requests <= 0) return;`
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

    // The row's operands, in the row's order. `Args::bind` inside
    // `hand::fire` checks them against the signature, so a drift between this
    // list and `ATTN_XQA_SIGS` is a refusal and not a shifted argument.
    //
    // `regions.page_bucket` occupies the `max_pages_per_seq` slot, and that is
    // the launcher's own substitution rather than this port's:
    // `attention_xqa.cu:320` passed `page_bucket` there. The kernel's
    // parameter is the ROW STRIDE it fills, and the caller's
    // `max_pages_per_seq` is only the number the stride was rounded up from.
    let values = [
        ArgValue::Ptr(kv_page_indices_d.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indptr_d.cast_mut().cast()),
        ArgValue::Ptr(kv_last_page_lens_d.cast_mut().cast()),
        ArgValue::Ptr(regions.page_table.cast()),
        ArgValue::Ptr(regions.seq_lens.cast()),
        ArgValue::I32(num_requests),
        ArgValue::I32(regions.page_bucket),
        ArgValue::I32(page_size),
    ];

    // `attention_xqa.cu:313`, transcribed:
    //
    //     build_xqa_metadata_kernel<<<num_requests, 128, 0, stream>>>(
    //
    // `num_requests` is `grid.x` because the kernel indexes the request by
    // `blockIdx.x`; see `METADATA_BLOCK` for why the width is a constant here
    // and not a rule there. `smem` is 0: the kernel declares no shared memory
    // at all — the sequence length is written by lane 0 out of registers, not
    // reduced.
    super::hand::fire(
        METADATA_SYMBOL,
        Launch {
            grid: [num_requests.unsigned_abs(), 1, 1],
            block: [METADATA_BLOCK, 1, 1],
            smem: 0,
        },
        &values,
        stream,
    );
    XqaPrepare::Prepared
}

// ===========================================================================
// The decode, which is the other five sixths of the family
// ===========================================================================

/// `attention_xqa.cu:198`'s `kXqaHeadDim`, and the only head width the lattice
/// is instantiated at (`-DHEAD_ELEMS=128`, `attention_xqa_gqa2.cu:25`).
pub const XQA_HEAD_DIM: i32 = 128;

/// `attention_xqa.cu:198`'s `kXqaPageSize` — the page size five of the six
/// lattice members are built for (`-DTOKENS_PER_PAGE=32`).
pub const XQA_PAGE_SIZE: i32 = 32;

/// `sizeof(SharedMem)` (`xqa/mha.cu:409`), measured out of NVRTC's PTX at
/// `compute_89` for every member of the lattice: **79,488 bytes, and the same
/// number for all of them.** HEAD_GRP_SIZE 2, 4, 5 and 8 at TOKENS_PER_PAGE
/// 32, and HEAD_GRP_SIZE 2 at TOKENS_PER_PAGE 16, all emit
/// `.global .align 4 .u32 pie_xqa_smem_size = 79488;`.
///
/// # This is half of what `configureKernel()` was
///
/// `xqa/mha.cu:2955` is a host static initializer that reads the device
/// constant and then opts the kernel in to it:
///
/// ```text
/// static uint32_t configureKernel() {
///   uint32_t size;
///   cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize));
///   cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size);
///   return size;
/// }
/// static uint32_t const hostSmemSize = configureKernel();   // mha.cu:2962
/// ```
///
/// **The opt-in half needs no code here at all.**
/// `kernels_cuda_new::runtime::module::raise_dynamic_smem_cap` is called by
/// `KernelModule::fire` whenever `Launch::smem` exceeds `DEFAULT_DYNAMIC_SMEM`
/// (48 KiB, `module.rs:353`), and it is keyed per `(CUdevice, CUfunction)`.
/// That is the hook north star §5 step 1 names, and putting this number in
/// [`XqaLaunch::smem`] is the whole of using it. It also discharges the
/// obligation `attention_xqa.hpp` records as OPEN — *"an undischarged
/// per-device `cudaFuncSetAttribute` under TP>1"*, §50.9 — because a per
/// `(device, function)` key is exactly the thing a per-process C++ static
/// initializer could not be.
///
/// **The read-back half is a constant here rather than a device read**, and
/// that is a compromise with a measurement behind it. Upstream's symbol is
/// `CUBIN_EXPORT __device__ constexpr uint32_t smemSize` (`xqa/mha.cu:409`),
/// and `constexpr` at namespace scope is internal linkage: the PTX says
/// `.global .align 4 .u32 smemSize = 79488;` with **no `.visible`**.
/// `csrc/src/attn/attention_xqa_mha.cuh` re-exports it as
/// `pie_xqa_smem_size` to get a name we control, and — measured — that does
/// not come out `.visible` either, with or without `const`, with or without
/// `nvrtcAddNameExpression("&pie_xqa_smem_size")` (which does return the
/// lowered name, rc = 0). Whether `cuModuleGetGlobal` resolves a
/// non-`.visible` `.global` needs a CUDA context to answer and the port was
/// written without one.
///
/// So: this constant is what the fire uses, `pie_xqa_smem_size` is the check
/// that would catch it drifting, and the check needs a `cuModuleGetGlobal`
/// accessor that `runtime::module` does not have. `fa2.rs:527`'s
/// `ECHO_TEMPLATE` wants the same accessor for the same reason and records
/// *"Nothing reads it yet."*
pub const XQA_SMEM_BYTES: u32 = 79_488;

/// `xqa/mha.cu:2760`'s `__launch_bounds__(256, nbCtaPerSM)`, derived rather
/// than read off a `<<<>>>`.
///
/// `launchMHAFlashInfer` declares its block as a NAMED `dim3`, which is
/// invisible to anything parsing the text between `<<<` and `>>>`:
///
/// ```text
/// xqa/mha.cu:76        ctaShapeInWarps = {4, 1, 2}
/// xqa/utils.cuh:256    warp_size = 32
/// xqa/mha.cu:~2999     dim3 dimCta{warp_size * ctaShapeInWarps.x,
///                                  ctaShapeInWarps.y, ctaShapeInWarps.z}
/// ```
///
/// so `{32 * 4, 1, 2}` = **(128, 1, 2)**, 256 threads, which is what the
/// `__launch_bounds__` on the kernel independently says.
pub const XQA_BLOCK: [u32; 3] = [128, 1, 2];

/// `xqa/mha.cu:96`'s `ctaTile.x`, the sequence-length step one CTA covers.
///
/// `warpTile = {64, roundUp(nbValidRows, 16)}` and
/// `ctaTile.x = warpTile.x * ctaShapeInWarps.x` = `64 * 4` = 256. It is the
/// denominator in the multi-block split below, so it is a geometry input and
/// not a comment.
const XQA_CTA_TILE_X: u32 = 256;

/// Which member of the lattice a shape selects.
///
/// The archive spelled this as five `detail::launch_attention_xqa_decode_bf16_*`
/// declarations and an `if`/`else if` chain over `head_group_ratio`
/// (`attention_xqa.cu:290-436`). It is an enum here because the answer is one
/// of a closed set and the C++ shape let a sixth arm be added without anyone
/// noticing the lattice had grown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XqaMember {
    /// `-DHEAD_GRP_SIZE=2 -DTOKENS_PER_PAGE=32`.
    Gqa2Page32,
    /// `-DHEAD_GRP_SIZE=2 -DTOKENS_PER_PAGE=16`.
    ///
    /// Unreachable today: [`gqa2_page16_enabled`] is `false`, so
    /// [`decode_supported`] refuses page 16 before this can be picked.
    Gqa2Page16,
    /// `-DHEAD_GRP_SIZE=4 -DTOKENS_PER_PAGE=32`.
    Gqa4Page32,
    /// `-DHEAD_GRP_SIZE=5 -DTOKENS_PER_PAGE=32`.
    Gqa5Page32,
    /// `-DHEAD_GRP_SIZE=8 -DTOKENS_PER_PAGE=32`, the Ampere/Ada body.
    Gqa8Page32,
    /// `-DHEAD_GRP_SIZE=8 -DTOKENS_PER_PAGE=32 -DUSE_SM90_MHA=1`.
    ///
    /// `attention_xqa_gqa8.cu` forwards to this when
    /// `current_device_major() >= 9`, so on every device that
    /// [`decode_supported`] admits, [`XqaMember::Gqa8Page32`] is in fact
    /// never the one that runs. Both are kept because the forward is the
    /// archive's structure and collapsing it would delete the record of a
    /// body that exists.
    Gqa8Page32Sm90,
}

impl XqaMember {
    /// The `extern "C"` device entry this member exports, after the
    /// `-Dkernel_mha=…` rename that `kernels_cuda_new::families::attn`'s
    /// `XQA_LATTICE` carries.
    ///
    /// The rename is not cosmetic. The archive renamed the HOST launcher six
    /// ways (`#define launchMHA launchMHA_xqa_gqa2_bf16_p32_h128`,
    /// `attention_xqa_gqa2.cu:32`) because six translation units defining the
    /// same static symbols were about to be linked into one archive. Under
    /// NVRTC there is no linker to collide in — but
    /// `kernels_cuda_new::unit::unit_of` resolves a symbol across the whole
    /// table, so six units all exporting `kernel_mha` are six rows nothing
    /// can tell apart. The collision moved; it did not go away.
    #[must_use]
    pub fn entry(self) -> &'static str {
        match self {
            Self::Gqa2Page32 => "kernel_mha_xqa_gqa2_bf16_p32_h128",
            Self::Gqa2Page16 => "kernel_mha_xqa_gqa2_bf16_p16_h128",
            Self::Gqa4Page32 => "kernel_mha_xqa_gqa4_bf16_p32_h128",
            Self::Gqa5Page32 => "kernel_mha_xqa_gqa5_bf16_p32_h128",
            Self::Gqa8Page32 => "kernel_mha_xqa_gqa8_bf16_p32_h128",
            Self::Gqa8Page32Sm90 => "kernel_mha_xqa_gqa8_sm90_bf16_p32_h128",
        }
    }

    /// The shape that picks this member, or `None` for a shape outside the
    /// lattice — `attention_xqa.cu:290-436`'s dispatch chain.
    ///
    /// `major` is the device's compute capability major, and it is a
    /// parameter rather than a query because `attention_xqa_gqa8.cu:96`'s
    /// forward to the Hopper body is a dispatch decision, not a device fact
    /// this function should be reaching out for.
    #[must_use]
    pub fn pick(head_group_ratio: i32, page_size: i32, major: i32) -> Option<Self> {
        match (head_group_ratio, page_size) {
            (2, 32) => Some(Self::Gqa2Page32),
            (2, 16) => Some(Self::Gqa2Page16),
            (4, 32) => Some(Self::Gqa4Page32),
            (5, 32) => Some(Self::Gqa5Page32),
            // `attention_xqa_gqa8.cu`: the gqa8 launcher forwards to the sm90
            // body on Hopper and above, and runs its own below.
            (8, 32) if major >= 9 => Some(Self::Gqa8Page32Sm90),
            (8, 32) => Some(Self::Gqa8Page32),
            _ => None,
        }
    }
}

/// `attention_xqa.cu:227`'s `xqa_gqa2_page16_enabled()`.
///
/// `false`, and it has always been `false`. The 16-token-page member of the
/// lattice is built and never selected; the flag is the switch that would
/// select it. Transcribed rather than dropped because a constant `false` in
/// Rust is the same claim the C++ made and deleting it would turn flipping a
/// flag back into a port.
#[must_use]
pub const fn gqa2_page16_enabled() -> bool {
    false
}

/// `attention_xqa.cu:217-242`'s `xqa_decode_bf16_supported`, transcribed.
///
/// ```text
/// if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return false;
/// const int ratio = num_q_heads / num_kv_heads;
/// if (!xqa_ratio_supported(ratio)) return false;
/// const bool page_supported =
///     page_size == kXqaPageSize ||
///     (ratio == 2 && page_size == 16 && xqa_gqa2_page16_enabled());
/// if (head_dim != kXqaHeadDim || !page_supported) return false;
/// if (window_left >= 0 || logits_soft_cap > 0.f) return false;
/// const float default_scale = 1.0f / std::sqrt((float)head_dim);
/// if (sm_scale > 0.f && std::abs(sm_scale - default_scale) > 1.0e-6f) return false;
/// return current_device_major() >= 9;
/// ```
///
/// # The SM90 floor is a deployment measurement, not a capability one
///
/// `attention_xqa.cu:237-240`, verbatim, because it is the sentence that
/// stops someone lowering the bound after finding that the code compiles:
///
/// > FlashInfer's public XQA wrapper only enables this path on SM90+. The
/// > Ampere/Ada csrc instantiations compile, but local SM89 TP2 serving runs
/// > can spin indefinitely after graph capture, so keep those devices on the
/// > regular FlashInfer decode path.
///
/// The NVRTC port re-confirms the first clause from the other side: every
/// non-Hopper member of the lattice compiles clean at `compute_89`, 0 errors.
/// *Compiling* was never the question.
///
/// # The scale test is an equality test wearing a tolerance
///
/// XQA folds `1/sqrt(head_dim)` into the kernel and takes `qScale` as a
/// separate multiplier that the launcher passes as `1.0`. So a caller-supplied
/// `sm_scale` is not applied — it is CHECKED, against the default, at 1e-6.
/// Anything else is a different kernel's job. `sm_scale <= 0` means "unset"
/// and passes.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn decode_supported(
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    major: i32,
) -> bool {
    if num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0 {
        return false;
    }
    let ratio = num_q_heads / num_kv_heads;
    // `xqa_ratio_supported` is the lattice's own membership test; asking
    // `XqaMember::pick` is the same question with one fewer place to be wrong.
    // `major` is not consulted for membership here — the 8-at-32 pair answers
    // either way — so the Hopper floor below is the only device gate.
    if XqaMember::pick(ratio, XQA_PAGE_SIZE, major).is_none() {
        return false;
    }
    let page_supported =
        page_size == XQA_PAGE_SIZE || (ratio == 2 && page_size == 16 && gqa2_page16_enabled());
    if head_dim != XQA_HEAD_DIM || !page_supported {
        return false;
    }
    if window_left >= 0 || logits_soft_cap > 0.0 {
        return false;
    }
    let default_scale = 1.0f32 / (head_dim as f32).sqrt();
    if sm_scale > 0.0 && (sm_scale - default_scale).abs() > 1.0e-6 {
        return false;
    }
    major >= 9
}

/// Why a decode enqueued nothing.
///
/// Separate from [`Decline`] rather than adding variants to it: the prepare
/// refuses for one reason and the decode refuses for four, and one enum
/// covering both would let a `match` on a prepare's answer look total while
/// silently admitting a decode's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeDecline {
    /// `attention_xqa.cu:290` — `if (num_requests <= 0) return;`
    ///
    /// The same legal no-op [`Decline::NoRequests`] catches, for the same
    /// reason: a zero `grid.z` reaching `KernelModule::fire` is
    /// `Error::Geometry`, which would turn a no-op into a fault.
    NoRequests,
    /// [`decode_supported`] said no.
    ///
    /// The C++ threw here (`attention_xqa.cu:288`,
    /// `throw std::runtime_error("xqa decode: unsupported shape")`), and the
    /// throw was load-bearing in one direction only: it forced the caller to
    /// have tested the same predicate first. A returned refusal keeps the
    /// force — [`XqaDecode`] is `#[must_use]` — without making a routing
    /// decision an exception.
    UnsupportedShape,
    /// The float workspace cannot hold the page table, the sequence lengths
    /// and XQA's scratch.
    ///
    /// `attention_xqa_gqa2.cu:139` —
    /// `throw std::runtime_error("xqa gqa2 decode: attention workspace too small")`.
    FloatWorkspaceTooSmall,
    /// The int workspace cannot hold `num_requests * num_kv_heads` semaphores.
    ///
    /// `attention_xqa_gqa2.cu:150` —
    /// `throw std::runtime_error("xqa gqa2 decode: semaphore workspace too small")`.
    SemaphoreWorkspaceTooSmall,
}

/// Everything a decode fire needs that is not a pointer, computed.
///
/// The five archive launchers each recomputed this and each recomputed it
/// identically; `attention_xqa.cu`'s own header calls the duplication out.
/// One value here is the same arithmetic done once.
#[derive(Debug, Clone, Copy)]
pub struct XqaLaunch {
    /// Which body runs, and therefore which symbol is fired.
    pub member: XqaMember,
    /// `dim3 dimGrid{nbSubSeqPerSeq, nbKHeads, batchSize}` — `xqa/mha.cu:2996`.
    ///
    /// Read from the surrounding code, not from between `<<<` and `>>>`:
    /// `launchMHAFlashInfer` builds a NAMED `dim3` and hands it to
    /// `cudaLaunchKernelEx`, so there is no inline geometry to parse.
    pub grid: [u32; 3],
    /// [`XQA_BLOCK`].
    pub block: [u32; 3],
    /// [`XQA_SMEM_BYTES`]. Over 48 KiB, which is what makes
    /// `KernelModule::fire` raise the cap.
    pub smem: u32,
    /// `page_bucket * page_size` — the `maxSeqLen` the launcher passes.
    ///
    /// `attention_xqa_gqa2.cu:176`, and note what it is NOT: the caller's
    /// `max_pages_per_seq * page_size`. The page table's row stride is the
    /// power-of-two bucket ([`page_bucket`]), the kernel strides by it, and
    /// the sequence length it is told about has to be the strided one.
    pub max_seq_len: u32,
    /// `num_requests * num_kv_heads` `u32`s in the int workspace, which the
    /// launcher zeroes before every fire (`attention_xqa_gqa2.cu:152`).
    pub semaphore_count: u32,
    /// `kv_stride_page` in ELEMENTS — `page_size * num_kv_heads * head_dim`.
    pub kv_stride_page: u64,
    /// `kv_stride_token` in elements — `num_kv_heads * head_dim`.
    pub kv_stride_token: u64,
    /// `kv_stride_head` in elements — `head_dim`.
    pub kv_stride_head: u64,
    /// `enable_pdl`.
    ///
    /// `current_device_major() >= 9` in five of the six launchers and an
    /// unconditional `true` in `attention_xqa_gqa8_sm90.cu` — which is the
    /// same predicate, since that body only runs on Hopper.
    ///
    /// It is the ONLY thing `xqa/hostUtils.h`'s `makeLaunchConfig` adds over
    /// a plain launch: a single
    /// `cudaLaunchAttributeProgrammaticStreamSerialization`. With PDL off, a
    /// bare `cuLaunchKernel` reproduces the launch exactly; with it on, the
    /// attribute has no `Launch` field to carry it and no `cuLaunchKernelEx`
    /// in `runtime::module` to set it.
    pub enable_pdl: bool,
}

/// What a decode did.
///
/// # `Launched` is deliberately absent
///
/// Because nothing can launch yet, and a variant named for an outcome the
/// code cannot produce is the kind of thing that gets wired up and then
/// discovered. [`plan_decode`] computes the whole host program — the gate,
/// the member, the carve, the geometry, the strides — and stops one step
/// short of the fire.
///
/// The device half is no longer what is missing. `csrc/vendor/xqa/` carries
/// the fifteen-file closure of upstream's `xqa/mha.cu` — as `xqa/mha.cuh`,
/// because `kernels-cuda-new` holds no translation units — and all five
/// non-Hopper members compile clean through it: rc = 0, one `.visible
/// .entry` per member under the renamed symbol [`XqaMember::entry`] returns,
/// measured with the tree's own `--fmad=false --prec-div=true
/// --prec-sqrt=true`.
///
/// Those three flags are also why this port must not claim archive parity.
/// The archive passes none of them, so nvcc built the same instantiations
/// under `--fmad=true`: it contracts multiply-adds and the JIT refuses to.
/// The JIT is the stricter of the two, and the two are not bit-identical by
/// construction (`new-horizon.md` §62.8).
///
/// Every `xqa/mha.cu:N` cited in this file is a line in UPSTREAM's copy,
/// which is the anchor that does not move; `csrc/vendor/xqa/mha.cuh`'s own
/// `// PIE:` header carries the offsets.
///
/// What is missing is the row, and it is missing for one reason:
///
/// * **A byte-buffer `ArgValue`.** `kernel_mha` takes
///   `KVCacheList<usePagedKVCache> const cacheList` **by value**
///   (`xqa/mha.cu:2757`); with `ENABLE_4BIT_KV_CACHE` off
///   (`xqa/mhaUtils.cuh:242-253`) that is four pointers and a `uint32_t` —
///   **40 bytes, 8-aligned**. `runtime::args::ArgValue` is
///   `Ptr/I32/U32/F32/Usize/I64/Bool/U8`. North star §3.2 names the gap:
///   *"a borrowed byte buffer, so by-value aggregates over 8 bytes … can
///   cross the JIT path."* Without it there is no `KernelSig` to write, and
///   `tests/units.rs:436` fails a unit that declares no rows — so the unit
///   cannot be enrolled either.
///
/// So `attention_xqa.cu` and its five siblings are **not deleted**. They hold
/// the only definition of `attn::attention_xqa_decode_bf16_prepared`, which
/// is a live `table::attn` row with `operands` and a live
/// `model-compiler/src/dsl.rs:6891` consumer; deleting them before the fire
/// exists would break the link, not migrate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum XqaDecode {
    /// The host program ran to completion and this is what a fire would need.
    ///
    /// Nothing has been enqueued on the stream.
    Planned(XqaLaunch),
    /// Nothing was computed and nothing would be enqueued.
    Declined(DecodeDecline),
}

/// The decode's host program — `attention_xqa_gqa2.cu:109-193` and its four
/// identical twins, done once.
///
/// `sm_count` is the device's `multiProcessorCount`
/// (`attention_xqa_gqa2.cu:67`'s `current_device_sm_count()`) and `major` its
/// compute capability major, both passed rather than queried: the archive
/// cached them in `thread_local`s because it had no other place to put a
/// device fact, and a function that takes them is one that can be reasoned
/// about without one.
///
/// # The multi-block split, cited
///
/// ```text
/// xqa/defines.h:101    ALLOW_MULTI_BLOCK_MODE defaults to true
/// xqa/mha.cu:~2990     nbSubSeqPerSeq = allowMultiBlockMode
///                        ? min(max(1u, multiProcessorCount / (batchSize * nbKHeads)),
///                              divUp(maxSeqLen, ctaTile.x))
///                        : 1
/// xqa/mha.cu:~2996     dim3 dimGrid{nbSubSeqPerSeq, nbKHeads, batchSize}
/// ```
///
/// The first term is *"how many CTAs can I afford per (request, kv head)"* and
/// the second is *"how many are there sequence to cover"*; the smaller wins.
/// `divUp` is `(a + b - 1) / b` (`xqa/utils.cuh`), and `ctaTile.x` is
/// [`XQA_CTA_TILE_X`].
///
/// # What is NOT reproduced, and why
///
/// `CUDA_CHECK(cudaPeekAtLastError())` after the launch. The archive's own
/// note on the prepare's move applies unchanged: a peek-and-throw in a fire
/// path attributes an unrelated async fault to the next kernel that happens
/// to run.
///
/// The `cudaMemsetAsync` that zeroes the semaphore bank
/// (`attention_xqa_gqa2.cu:152`) is NOT done here either, and that one is a
/// real omission rather than a decision: it must happen on the stream before
/// the fire, and this function does not touch the stream. [`XqaLaunch`]
/// reports [`XqaLaunch::semaphore_count`] so the caller that does the fire
/// does the memset, in the one place both are ordered.
#[allow(clippy::too_many_arguments)]
pub fn plan_decode(
    workspace: AttentionWorkspaceView,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    max_pages_per_seq: i32,
    sm_scale: f32,
    sm_count: i32,
    major: i32,
) -> XqaDecode {
    // `attention_xqa.cu:285` — the gate runs before the empty-set test,
    // because a shape that is wrong is wrong whether or not anything asked.
    // `window_left` and `logits_soft_cap` are the two the caller pinned:
    // `/*window_left=*/-1, /*logits_soft_cap=*/0.f` at `attention_xqa.cu:287`.
    if !decode_supported(
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        -1,
        0.0,
        sm_scale,
        major,
    ) {
        return XqaDecode::Declined(DecodeDecline::UnsupportedShape);
    }
    if num_requests <= 0 {
        return XqaDecode::Declined(DecodeDecline::NoRequests);
    }

    let ratio = num_q_heads / num_kv_heads;
    let Some(member) = XqaMember::pick(ratio, page_size, major) else {
        return XqaDecode::Declined(DecodeDecline::UnsupportedShape);
    };

    // The same carve the prepare wrote, read back rather than recomputed —
    // the page table the decode reads has to be the one the prepare filled,
    // and two copies of this arithmetic is exactly what the archive had.
    let regions = carve(workspace, num_requests, max_pages_per_seq);
    let end = workspace.float_buffer.addr() + workspace.float_bytes;
    if regions.scratch.addr() >= end {
        return XqaDecode::Declined(DecodeDecline::FloatWorkspaceTooSmall);
    }

    let semaphore_count = num_requests.unsigned_abs() * num_kv_heads.unsigned_abs();
    if semaphore_count as usize * size_of::<u32>() > workspace.int_bytes {
        return XqaDecode::Declined(DecodeDecline::SemaphoreWorkspaceTooSmall);
    }

    let batch = num_requests.unsigned_abs();
    let kv_heads = num_kv_heads.unsigned_abs();
    let max_seq_len = regions.page_bucket.unsigned_abs() * page_size.unsigned_abs();

    // `xqa/mha.cu:~2990`, transcribed. `ALLOW_MULTI_BLOCK_MODE` is
    // `xqa/defines.h:101`'s default and no unit overrides it, so the ternary's
    // false arm is unreachable and is not written.
    let afford = (sm_count.unsigned_abs() / (batch * kv_heads)).max(1);
    let need = max_seq_len.div_ceil(XQA_CTA_TILE_X);
    let nb_sub_seq_per_seq = afford.min(need);

    XqaDecode::Planned(XqaLaunch {
        member,
        grid: [nb_sub_seq_per_seq, kv_heads, batch],
        block: XQA_BLOCK,
        smem: XQA_SMEM_BYTES,
        max_seq_len,
        semaphore_count,
        // `attention_xqa_gqa2.cu:168-174` — all three in ELEMENTS, which is
        // what the archive computed and what `xqa/mha.cu`'s
        // `stride_X_in_heads = kv_stride_X / (validElemsPerHead /
        // CacheElemConverter::ElemsPerContainer)` then divides down. With
        // `CACHE_ELEM_ENUM=0` the container is one element and the division is
        // by `head_dim`.
        kv_stride_page: page_size as u64 * num_kv_heads as u64 * head_dim as u64,
        kv_stride_token: num_kv_heads as u64 * head_dim as u64,
        kv_stride_head: head_dim as u64,
        enable_pdl: major >= 9,
    })
}
