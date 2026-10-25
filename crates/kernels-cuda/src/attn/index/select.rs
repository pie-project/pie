//! A scored index's ranking with a row's keys spread over a cluster of CTAs
//! (`attn/index/select.cu`): the radix select's picks in its order, for a
//! decode's few long rows. Arch-generic: clusters where the chip has them
//! (sm_90 up), a CTA a row below.
#![allow(unsafe_code)]

use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Kernel, Launch, kernel, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const SELECT: &str = "attn/index/select.cu";

/// `pie::attn::index::split::kThreads`.
const THREADS: u32 = 512;

/// `split::kMaxRanks`: the widest cluster (B200 seats 16 one-block SMs one).
const MAX_RANKS: u32 = 16;

/// `split::kHeader`: a CTA's shared memory ahead of its chunk of order bits
/// (the cluster's three histograms of 2048 bins, its own, the scans'
/// counts), all of it dynamic.
const HEADER: u32 = 34_032;

/// The fewest scores a CTA of a spread row takes: below it the cluster's
/// barriers cost more than the CTA saves.
const MIN_CHUNK: u32 = 1024;

/// The narrowest rows spread over a cluster: below it one CTA ranks a row
/// as fast as a cluster's barriers let one (B200: 8192 scores 9.8 us alone,
/// 9.6 over eight; 16384 14.0 alone, 10.1 over eight).
const SPREAD_FROM: u32 = 16_384;

/// The narrowest rows (the scores scratch's stride) ranked here: below it a
/// block a row is as fast.
const SPLIT_FROM: u32 = 4096;

/// `pie::attn::index::SplitParams`, field for field.
#[repr(C)]
struct SplitParams {
    qo_indptr: u64,
    page_indptr: u64,
    last_page_lens: u64,
    scores: u64,
    win: u64,
    requests: i32,
    page_size: i32,
    score_stride: i32,
    top_k: i32,
    ratio: i32,
    rows: i32,
    ranks: i32,
}

/// The rows a scorer landed in `scores` ranked with each row's keys over a
/// cluster of CTAs held in their shared memory (`index_select_split`):
/// `Ok(false)` where the ranking is not the radix select's
/// (`kernel.sdpa.index_radix` off), the fire has more rows than
/// `kernel.sdpa.index_split_until`, its rows are narrower than
/// `SPLIT_FROM`, or a row's keys outgrow the widest cluster's memory, and
/// the caller ranks a block a row.
#[allow(clippy::too_many_arguments)]
pub(super) fn ranked_split(
    ctx: &Ctx,
    q: RaggedTensor,
    keys: &KvPool,
    top_k: i32,
    ratio: i32,
    score_stride: i32,
    scores: u64,
    num_requests: i32,
    selection: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "attn.index_topk";
    let rows = selection.rows;
    if !ctx.tuning().index_radix || rows > ctx.tuning().index_split_until || rows == 0 {
        return Ok(false);
    }
    let stride = score_stride.unsigned_abs();
    if stride < SPLIT_FROM {
        return Ok(false);
    }
    let entry = kernel(SELECT, "::pie::attn::index::index_select_split");
    let Some(ranks) = ranks(ctx, entry, rows, stride) else {
        return Ok(false);
    };
    let params = SplitParams {
        qo_indptr: q.indptr.ptr,
        page_indptr: keys.page_indptr.ptr,
        last_page_lens: keys.last_page_lens.ptr,
        scores,
        win: match ctx.stage() {
            ArgValue::Ptr(at) => at,
            _ => 0,
        },
        requests: num_requests,
        page_size: keys.page_size,
        score_stride,
        top_k,
        ratio,
        rows: stated(OP, rows)?,
        ranks: stated(OP, ranks)?,
    };
    let launch = Launch::grid([rows * ranks, 1, 1], [THREADS, 1, 1])
        .smem(HEADER + chunk(stride, ranks) * 4)
        .programmatic(ctx.tuning().pdl);
    let launch = if ranks > 1 {
        launch.cluster(ranks)
    } else {
        launch
    };
    ctx.fire(
        OP,
        Fire::at(entry).apply(launch),
        &[ArgValue::of(&params), selection.arg()],
    )?;
    Ok(true)
}

/// A CTA's scores of a `stride`-wide row over `ranks`, whole quads.
const fn chunk(stride: u32, ranks: u32) -> u32 {
    stride.div_ceil(ranks).next_multiple_of(4)
}

/// The CTAs a row, a power of two (the histograms' bins split evenly over
/// the ranks): the widest whose clusters the device seats every row's of at
/// once with no CTA under `MIN_CHUNK` scores (one CTA a row narrower than
/// `SPREAD_FROM`), and no narrower than a chunk's shared memory needs;
/// `None` where even the widest cluster's does not hold the row.
fn ranks(ctx: &Ctx, entry: &'static Kernel, rows: u32, stride: u32) -> Option<u32> {
    let room = ctx
        .smem_per_block_optin()
        .unwrap_or(227 * 1024)
        .saturating_sub(HEADER);
    let clusters = crate::jit::compute_capability_major().is_some_and(|major| major >= 9);
    let widest = if clusters { MAX_RANKS } else { 1 };
    let mut least = 1;
    while chunk(stride, least) * 4 > room {
        least *= 2;
        if least > widest {
            return None;
        }
    }
    if least > 1 && seated(entry, least) == 0 {
        return None;
    }
    let mut ranks = least;
    while stride >= SPREAD_FROM
        && ranks * 2 <= widest
        && chunk(stride, ranks * 2) >= MIN_CHUNK
        && rows <= seated(entry, ranks * 2)
    {
        ranks *= 2;
    }
    Some(ranks)
}

/// The clusters of `ranks` CTAs of `entry` the device seats at once, 0
/// where the driver cannot say; asked once a width. A CTA's 512 threads
/// hold a multiprocessor alone, so a chunk's memory changes nothing.
fn seated(entry: &'static Kernel, ranks: u32) -> u32 {
    use cudarc::driver::sys as dr;
    use std::sync::Mutex;

    static SEATED: Mutex<Vec<(usize, u32, u32)>> = Mutex::new(Vec::new());
    let Ok(resolved) = entry.resolve() else {
        return 0;
    };
    let key = resolved.function.addr();
    let mut seen = SEATED.lock().expect("seated lock poisoned");
    if let Some(&(_, _, n)) = seen.iter().find(|&&(f, r, _)| f == key && r == ranks) {
        return n;
    }
    // SAFETY: `CUlaunchAttribute` is a C struct of an enum id and a
    // plain-data union; all-zero bytes are a valid value.
    let mut attr: dr::CUlaunchAttribute = unsafe { core::mem::zeroed() };
    attr.id = dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim.x = ranks;
    attr.value.clusterDim.y = 1;
    attr.value.clusterDim.z = 1;
    let config = dr::CUlaunchConfig {
        gridDimX: ranks,
        gridDimY: 1,
        gridDimZ: 1,
        blockDimX: THREADS,
        blockDimY: 1,
        blockDimZ: 1,
        sharedMemBytes: HEADER,
        hStream: core::ptr::null_mut(),
        attrs: &raw mut attr,
        numAttrs: 1,
    };
    let mut clusters: core::ffi::c_int = 0;
    // SAFETY: `resolved.function` is a handle the cache keeps loaded for the
    // process; `config` and its attribute are locals that outlive the calls,
    // which launch nothing. Past the portable 8 a cluster is the function's
    // to allow, as a launch allows it.
    let code = unsafe {
        if ranks > 8 {
            dr::cuFuncSetAttribute(
                resolved.function,
                dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            );
        }
        dr::cuOccupancyMaxActiveClusters(&raw mut clusters, resolved.function, &raw const config)
    };
    let n = if code == dr::CUresult::CUDA_SUCCESS {
        u32::try_from(clusters).unwrap_or(0)
    } else {
        0
    };
    seen.push((key, ranks, n));
    n
}
