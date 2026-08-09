//! The fused LM-head GEMV + argmax: the host half of `sample/argmax.cu`, in
//! Rust.
//!
//! Ports `kernels::sample::lm_head_gemv_argmax_int8` — the file's last
//! launcher, and the one `families::sample`'s header spent a paragraph
//! refusing to state as a rule. The launcher, its header
//! (`sample/argmax.hpp`), its `CMakeLists.txt` entry and the whole
//! `csrc/src/sample/` directory are DELETED; the twelve `__global__`
//! templates are `kernels-cuda-new`'s `sample/argmax` unit, which NVRTC
//! compiles.
//!
//! # Why this one is the archetype
//!
//! The owner's principle has two clauses and this body is both at once:
//!
//! > Every CUDA kernel is compiled by NVRTC. Where host code is needed to
//! > compose kernels — because kernels produce intermediate results, or
//! > because device-specific tuning is involved — that host code is all Rust.
//!
//! *Intermediate results*: the first kernel writes one packed `(value,
//! token)` pair per tile per row into a scratch buffer, and the second folds
//! the tiles. Nothing in `table::sample`'s operand list mentions that buffer.
//! *Device-specific tuning*: `grid.x` is `min(num_sms * 2, ceil(vocab / 8))`,
//! read straight off `cudaDevAttrMultiProcessorCount`.
//!
//! # The five things no row says, and where each one went
//!
//! `kernels_cuda_new::families::sample`'s header lists them. They are all
//! still true — none is retracted by rowing the two kernels, because a row is
//! a NAME and an operand list and never a geometry:
//!
//! | the fact | `argmax.cu` | here |
//! | --- | --- | --- |
//! | grid.x from an occupancy query | `:101-107` | `blocks_x` |
//! | a 2-D grid over (blocks, rows) | `:121` | the first `Launch` below |
//! | dynamic shmem `hidden * 4` | `:108-109` | the same `Launch` |
//! | a `static` scratch that grows | `:111-119` | `PAIRS` |
//! | two kernels per call | `:123`, `:136` | [`lm_head_gemv_argmax_int8`] |
//!
//! Both launches build a [`Launch`] by hand and fire it through
//! `KernelModule::fire`, which is what [`super::attn_score`] and
//! [`super::gemv`] do for their own unstatable grids. **No `LaunchRule`
//! variant was added**: `new-horizon.md` §10.5's bar is that a rule must
//! serve more kernels than the one that wants it, and a grid whose extent
//! comes from a device query serves exactly one.
//!
//! # The scratch, reproduced rather than improved — and its inherited defect
//!
//! `argmax.cu:113-119` is a function-local `static device::u64*` with a
//! `static usize` capacity beside it, `cudaFree`d and re-`cudaMalloc`d
//! whenever `num_blocks_x * num_rows` outgrows it. `PAIRS` is that, as a
//! value, with the allocation under a lock so the GROWTH is not a data race.
//!
//! **The buffer is still shared by every caller, and that is a defect this
//! port inherits deliberately.** Two worker threads firing this row on two
//! streams write the same scratch, and holding a lock across the launches
//! does not fix it: a launch is asynchronous, so the second thread's GEMV can
//! land in the buffer before the first thread's selector has read it, and the
//! result is a plausible token id for the wrong row. The C++ had exactly this
//! hazard and this port does not change it, because
//! [`super::gemv`]'s rule applies — *"the port's first duty is to reproduce
//! today's launches rather than to improve them"*, and a port that changes
//! the arithmetic cannot be A/B'd against the arithmetic it replaced.
//!
//! What closes it, when someone wants to: the scratch belongs in the fire's
//! [`super::scratch`], which is already per-fire and already pooled for the
//! address-identity reason a recorded graph needs. That is a change to where
//! a buffer lives, not to what the kernels compute, and it can be measured
//! against this.
//!
//! # A failure is a refusal, never a fallback
//!
//! `argmax.cu` ignored `cudaMalloc`'s return code and launched over whatever
//! `s_partial_pairs` held — a null pointer on the first failure. This panics.
//! It also panics on a failed SM query, where [`super::gemv`] defaults: that
//! module can answer 4 for a failed compute-capability read because both arms
//! compute the same thing and one is slower, whereas here the queried number
//! IS `grid.x` and IS the `num_blocks_x` operand the kernel strides the vocab
//! by. A guess covers a subset of the vocabulary and reports the argmax of
//! it, which is a wrong answer wearing a right answer's shape.
//!
//! `execution::WALKED`'s entry for this symbol states all four refusals in
//! the launcher's own words.

use std::sync::Mutex;

use kernels_cuda_new::runtime::{ArgValue, Args, Launch, Stream, cache};

/// Threads per block of the fused GEMV — `sample/argmax.cuh:153-154`'s
/// `GEMV_WARPS = 8` times a warp, which the header spells
/// `GEMV_BLOCK_DIM = GEMV_WARPS * 32`.
///
/// Load-bearing twice, which is why it is derived from [`GEMV_WARPS`] rather
/// than written as 256: the kernel stages the hidden vector with `for (i =
/// threadIdx.x; i < hidden; i += GEMV_BLOCK_DIM)`, so a narrower block leaves
/// the tail of `sh_hidden` uninitialised and every dot product reads it.
const GEMV_BLOCK_DIM: u32 = GEMV_WARPS * 32;

/// Vocab rows a block covers per grid step — `sample/argmax.cuh:153`.
///
/// Also the divisor in `min_blocks_x = ceil(vocab / GEMV_WARPS)`: one warp
/// per vocab row, so the smallest grid that covers the vocab in one step is
/// the row count divided by the warps in a block. A grid computed with one
/// number and a kernel striding by another would either skip rows or repeat
/// them, and the kernel's `for (v = blockIdx.x * GEMV_WARPS + warp; v < vocab;
/// v += num_blocks_x * GEMV_WARPS)` cannot tell the difference.
const GEMV_WARPS: u32 = 8;

/// Resident blocks per SM the launcher aims for — `argmax.cu:103`'s
/// `constexpr int kBlocksPerSm = 2`.
///
/// A persistent-block tuning constant with one reader, which is the shape a
/// tuning constant should have. It is not an occupancy guarantee: the kernel
/// asks for `hidden * sizeof(float)` of dynamic shared memory, so at a wide
/// hidden size two blocks per SM may not fit and the hardware simply runs
/// fewer. The grid is a bound on how much work is resident, not a promise.
const BLOCKS_PER_SM: u32 = 2;

/// Threads per block of the selector — `argmax.cu:134`'s `dim3
/// sel_block(128)`.
///
/// A plain elementwise fold over rows, one thread per row. 128 rather than
/// the 256 every other pointwise launcher in this tree uses, and it is
/// transcribed rather than harmonised: `num_rows` is a batch size, so the
/// grid is one or two blocks either way and the difference is unmeasurable —
/// which makes changing it a diff with no argument behind it.
const SELECT_BLOCK: u32 = 128;

/// The producing kernel's row in `kernels_cuda_new::families::sample`.
const GEMV_SYMBOL: &str = "sample::lm_head_gemv_argmax_int8_bf16";

/// The folding kernel's row.
const SELECT_SYMBOL: &str = "sample::select_lm_head_argmax_pairs";

/// The growable pair scratch — `argmax.cu:113-114`'s two function-local
/// `static`s, as one value.
///
/// The pointer is kept as a `usize` because a `*mut` is not `Send` and this
/// is reachable from every worker thread, exactly as the C++ `static` was.
/// That is a statement about the ADDRESS being shared, not a claim that
/// sharing it is safe — see the module docs, which name the hazard the C++
/// had and this keeps.
///
/// Never freed at process exit, like the C++: the allocation is one buffer
/// whose lifetime is the process's, and a `Drop` that ran `cudaFree` during
/// static destruction would race the runtime's own teardown.
static PAIRS: Mutex<Pairs> = Mutex::new(Pairs { ptr: 0, cap: 0 });

/// [`PAIRS`]' two numbers.
struct Pairs {
    /// The device address, or 0 before the first allocation.
    ptr: usize,
    /// Capacity in `u64` PAIRS, not bytes — `argmax.cu:114`'s `s_pairs_cap`,
    /// which is compared against `pairs_elems` and never against a byte
    /// count.
    cap: usize,
}

/// `grid.x` for the fused GEMV — `argmax.cu:101-107`, transcribed.
///
/// ```text
/// cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
/// constexpr int kBlocksPerSm = 2;
/// const int max_blocks_x = num_sms * kBlocksPerSm;
/// const int min_blocks_x = (vocab + device::GEMV_WARPS - 1) / device::GEMV_WARPS;
/// const int num_blocks_x = std::min(max_blocks_x, min_blocks_x);
/// ```
///
/// The `min` is what makes the blocks persistent: enough to fill the machine,
/// never more than there is vocabulary for. It is also the kernel's
/// `num_blocks_x` operand, which is why this number is computed once and used
/// twice — a grid that disagreed with the operand would stride past the end
/// of the vocab or stop short of it, silently either way.
///
/// # The SM count is cached, and the query is this driver's own
///
/// [`crate::device::Device::sm_count`] rather than a second
/// `cudaDeviceGetAttribute` written here, which is the choice
/// [`super::gemv`]'s `unroll_depth` made and for the same reason. The C++ asked
/// on every call and passed ordinal `0` unconditionally; this asks once per
/// process, on the CURRENT device, and caches. On a single-GPU process those
/// are the same number. On a multi-GPU one the C++ was reading device 0's SM
/// count to size a grid for whatever device the stream belonged to — a bug
/// that happens not to matter because every GPU in a node is the same part,
/// and it is not reproduced.
///
/// # Panics
///
/// If the device cannot be bound or the attribute cannot be read. See the
/// module docs for why this is not a defaulted value.
fn blocks_x(vocab: i32) -> u32 {
    static SMS: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    let num_sms = *SMS.get_or_init(|| {
        use cudarc::runtime::sys as rt;

        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live, writable out-parameter for the call.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        assert!(
            code == rt::cudaError::cudaSuccess,
            "sample::lm_head_gemv_argmax_int8: cudaGetDevice failed ({code:?}), so the SM \
             count that sizes this launch cannot be read. There is no safe default: the \
             number is the grid AND the operand the kernel strides the vocab by"
        );
        let device = crate::device::Device::bind(ordinal).unwrap_or_else(|why| {
            panic!(
                "sample::lm_head_gemv_argmax_int8: could not bind device {ordinal} to read \
                 its SM count: {why}"
            )
        });
        device.sm_count().unwrap_or_else(|why| {
            panic!(
                "sample::lm_head_gemv_argmax_int8: cudaDevAttrMultiProcessorCount failed on \
                 device {ordinal}: {why}"
            )
        })
    });
    let max_blocks_x = num_sms.unsigned_abs() * BLOCKS_PER_SM;
    let min_blocks_x = vocab.unsigned_abs().div_ceil(GEMV_WARPS);
    max_blocks_x.min(min_blocks_x).max(1)
}

/// Greedy decode straight off an int8 LM head: `token_ids[r] = argmax_v
/// (hidden[r] . W[v]) * scale_inv[v]`.
///
/// `hidden_states` is bf16 `[num_rows, hidden]`, `lm_head_weight` is int8
/// `[vocab, hidden]` row-major, `scale_inv` is one fp32 dequant scale per
/// vocab row, and `token_ids` receives one i32 per row. Produces TOKEN IDS
/// and never materialises the vocab-wide logit row, which is why
/// `table::sample` states it as its own row rather than as an `lm_head`
/// followed by an argmax.
///
/// # Panics
///
/// On a failed SM query, a failed scratch allocation, an NVRTC compile
/// failure, or a disagreement between this call and the rows in
/// `kernels_cuda_new::families::sample`. See the module docs.
///
/// # Safety
///
/// Every pointer must address live device memory of the extents `num_rows`,
/// `hidden` and `vocab` describe, `token_ids` must be writable for `num_rows`
/// i32, and `stream` must be live across both launches — the obligations the
/// caller met when this was a `pie_k_` shim call
/// handing the stream to two `<<<>>>`.
pub unsafe fn lm_head_gemv_argmax_int8(
    hidden_states: *const std::ffi::c_void,
    lm_head_weight: *const i8,
    scale_inv: *const f32,
    token_ids: *mut i32,
    num_rows: i32,
    hidden: i32,
    vocab: i32,
    stream: *mut std::ffi::c_void,
) {
    // `argmax.cu:99`. A return and not a panic: an empty fire is a real
    // thing a batch produces and it was never an error.
    if num_rows <= 0 || hidden <= 0 || vocab <= 0 {
        return;
    }

    let num_blocks_x = blocks_x(vocab);
    let rows = num_rows.unsigned_abs();

    // `argmax.cu:111-119`. `pairs_elems` counts PAIRS; the allocation
    // multiplies by 8 and the capacity does not, exactly as `s_pairs_cap`
    // did.
    let pairs_elems = num_blocks_x as usize * rows as usize;
    let mut pairs = PAIRS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if pairs_elems > pairs.cap {
        use cudarc::runtime::sys as rt;

        if pairs.ptr != 0 {
            // SAFETY: the address came from `cudaMalloc` below and nothing
            // else frees it.
            let _ = unsafe { rt::cudaFree(pairs.ptr as *mut std::ffi::c_void) };
            pairs.ptr = 0;
            pairs.cap = 0;
        }
        let bytes = pairs_elems * std::mem::size_of::<u64>();
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `p` is a live, writable out-parameter.
        let code = unsafe { rt::cudaMalloc(std::ptr::from_mut(&mut p), bytes) };
        assert!(
            code == rt::cudaError::cudaSuccess && !p.is_null(),
            "sample::lm_head_gemv_argmax_int8: cudaMalloc({bytes}) for the pair scratch \
             failed ({code:?}). The C++ ignored this return code and launched over a null \
             pointer; refusing here is the difference between a diagnosable failure and a \
             token id read out of unwritten memory"
        );
        pairs.ptr = p as usize;
        pairs.cap = pairs_elems;
    }
    let partial_pairs = pairs.ptr as *mut std::ffi::c_void;

    // `argmax.cu:121-132`:
    //
    //     dim3 grid(num_blocks_x, num_rows);
    //     dim3 block(device::GEMV_BLOCK_DIM);
    //     device::lm_head_gemv_argmax_int8<device::bf16>
    //         <<<grid, block, shmem_bytes, stream>>>(
    //
    // with `shmem_bytes = hidden * sizeof(float)` from `:108-109` — the
    // staging buffer for one row of the hidden vector, which the kernel
    // declares `extern __shared__` and therefore cannot size itself.
    let smem = (hidden.unsigned_abs()).saturating_mul(4);
    let launch = Launch {
        grid: [num_blocks_x, rows, 1],
        block: [GEMV_BLOCK_DIM, 1, 1],
        smem,
    };
    // The row's operands, in the row's order —
    // `families::sample::ARGMAX_SIGS[3]`. `Args::bind` checks them against
    // the signature, so a drift between this list and that row is a refusal
    // and not a shifted argument.
    let values = [
        ArgValue::Ptr(hidden_states.cast_mut()),
        ArgValue::Ptr(lm_head_weight.cast_mut().cast()),
        ArgValue::Ptr(scale_inv.cast_mut().cast()),
        ArgValue::Ptr(partial_pairs),
        ArgValue::I32(num_rows),
        ArgValue::I32(hidden),
        ArgValue::I32(vocab),
        ArgValue::I32(i32::try_from(num_blocks_x).unwrap_or(i32::MAX)),
    ];
    fire(GEMV_SYMBOL, launch, &values, stream);

    // `argmax.cu:134-137`:
    //
    //     dim3 sel_block(128);
    //     dim3 sel_grid((num_rows + sel_block.x - 1) / sel_block.x);
    //     device::select_lm_head_argmax_pairs<<<sel_grid, sel_block, 0, stream>>>(
    //         s_partial_pairs, token_ids, num_rows, num_blocks_x);
    //
    // `num_blocks_x` arrives as `num_tiles`: the same number, named for what
    // it means to the reader instead of for what it meant to the producer.
    let launch = Launch {
        grid: [rows.div_ceil(SELECT_BLOCK), 1, 1],
        block: [SELECT_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(partial_pairs),
        ArgValue::Ptr(token_ids.cast()),
        ArgValue::I32(num_rows),
        ArgValue::I32(i32::try_from(num_blocks_x).unwrap_or(i32::MAX)),
    ];
    // `partial_pairs` was written by the launch above, on this stream, so the
    // ordering that makes it readable here is the stream's.
    fire(SELECT_SYMBOL, launch, &values, stream);

    drop(pairs);
}

/// Resolve one row through its unit and launch it, panicking on any drift.
///
/// [`super::gemv`]'s private `fire`, verbatim, for its reasons: a unit
/// that will not compile, or a row this file names and the kernel table does
/// not, is a build that disagrees with itself and there is nothing to fall
/// back to. Duplicated rather than shared because the two modules are
/// independent ports and a helper hoisted into `fire/mod.rs` would be a third
/// place to look for four lines.
///
/// # Panics
///
/// Every failure on this path is drift between this driver and its kernel
/// table, or a unit that will not compile. None of them may be answered with
/// a different kernel.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
fn fire(
    symbol: &'static str,
    launch: Launch,
    values: &[ArgValue],
    stream: *mut std::ffi::c_void,
) {
    let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
    };
    let module = match cache::module(index, unit) {
        Ok(module) => module,
        Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
    };
    let mut args = match Args::bind(sig, values) {
        Ok(args) => args,
        Err(why) => panic!("{symbol}: {why}"),
    };
    // SAFETY: the caller holds the fire's stream live across the launch.
    let stream = unsafe { Stream::from_runtime(stream) };
    if let Err(why) = module.fire(sig, launch, &mut args, stream) {
        panic!("{symbol}: {why}");
    }
}
