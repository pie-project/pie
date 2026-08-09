//! `DecodePlan`: how a batch of one-token queries is spread over CTAs.
//!
//! # The decision it makes
//!
//! A decode step has one QO row per request and a long KV per request, so the
//! only parallelism beyond *(request × KV head)* is splitting KV. Whether to
//! split is an occupancy question: if `batch_size * num_kv_heads` already fills
//! the grid, splitting buys nothing and costs a merge pass, so upstream does
//! not. If it does not fill the grid, it binary-searches the smallest KV chunk
//! size (in **pages**, not tokens) whose resulting work-item count still fits
//! — the smallest chunk that fills the machine, not the smallest chunk.
//!
//! # What is device-dependent here, and what is not
//!
//! `BatchDecodeWithPagedKVCacheWorkEstimationDispatched` computes a shared
//! memory size from the head dimension and the KV element type, asks
//! `cudaOccupancyMaxActiveBlocksPerMultiprocessor` **about the decode kernel**
//! how many CTAs of that size fit on an SM, and multiplies by the SM count.
//! Everything after that product is integer arithmetic on page counts.
//!
//! So the port takes the product. That is not a shortcut: the occupancy of a
//! kernel is a property of the compiled cubin — its register count and its
//! static shared memory — which this crate does not have and, once the kernel
//! is JIT-compiled, will not have until run time. Splitting the function there
//! puts every input this module cannot compute on the caller's side of the line
//! and leaves this side testable without a GPU. The dispatch's `smem_size`
//! computation is therefore **deliberately not ported**: it is an input to a
//! query we do not make.
//!
//! `gdy` is ported, because it is not device-dependent at all —
//! `num_qo_heads / GROUP_SIZE`, which is the KV head count.

use super::alloc::{AlignedAllocator, Staging};
use super::arith::{ceil_div_i32_in_u32, ceil_div_u32, cuda_max_u32_i32};
use super::info::DecodePlanInfo;
use super::{Error, Plan, Sizes, Workspace};

/// The batch a decode plan is built for.
///
/// `kv_indptr` is the **page** indptr: `kv_indptr[i + 1] - kv_indptr[i]` is
/// request `i`'s page count, and everything the estimator does is in pages
/// until the very last line, which multiplies by `page_size` to publish a
/// chunk size in tokens.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// `batch_size + 1` page offsets.
    pub kv_indptr: &'a [i32],
    /// Requests in the batch.
    pub batch_size: u32,
    /// Query/output heads.
    pub num_qo_heads: u32,
    /// The GQA group size the kernel was dispatched for; `num_kv_heads =
    /// num_qo_heads / gqa_group_size`, and that quotient is `gridDim.y`.
    pub gqa_group_size: u32,
    /// Tokens per page.
    pub page_size: u32,
    /// Per-head feature width, which only sizes the partial-output carve.
    pub head_dim: u32,
    /// Whether this plan will be captured into a CUDA graph, which fixes the
    /// grid and makes the padding — and the valid mask — necessary.
    pub enable_cuda_graph: bool,
}

impl Request<'_> {
    fn pages(&self, i: usize) -> i32 {
        // `int32_t` subtraction, wrap included: upstream's difference of two
        // `IdType`s, which overflows into undefined behaviour rather than a
        // panic. Nothing in this crate should abort on a malformed indptr that
        // the C++ merely mis-plans.
        self.kv_indptr[i + 1].wrapping_sub(self.kv_indptr[i])
    }

    fn check(&self) -> Result<(), Error> {
        let needed = self.batch_size as usize + 1;
        if self.kv_indptr.len() < needed {
            return Err(Error::IndptrTooShort {
                array: "kv_indptr",
                needed,
                got: self.kv_indptr.len(),
            });
        }
        Ok(())
    }
}

/// What the work estimator decided, before any buffer is carved.
///
/// Upstream returns this through five out-parameters and a `cudaError_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkEstimate {
    /// Whether KV is partitioned across work items.
    pub split_kv: bool,
    /// `num_blocks_per_sm * num_sm`, passed through unchanged so a caller can
    /// see what the rest was derived from.
    pub max_grid_size: u32,
    /// The KV chunk size **in pages** — upstream calls this
    /// `max_num_pages_per_batch` in the estimator and `kv_chunk_size_in_pages`
    /// at the call site, and they are the same number.
    pub kv_chunk_size_in_pages: u32,
    /// Work items the split produced.
    pub new_batch_size: u32,
    /// `gridDim.y`: the KV head count.
    pub gdy: u32,
}

/// `PartitionPagedKVCacheBinarySearchMinNumPagePerBatch`.
///
/// Binary-searches the smallest pages-per-chunk whose work-item count fits the
/// grid, then recomputes the count at that chunk size. The recomputation is not
/// redundant: the loop's last evaluated `new_batch_size` belongs to whichever
/// `mid` was tried last, which is not necessarily `low`.
///
/// The `std::max(elem, 1)` in the recomputation and the *absence* of it in the
/// loop is upstream's, and it matters for a request with zero pages: during the
/// search it contributes nothing, afterwards it contributes one work item. That
/// asymmetry is why an all-empty batch still gets `new_batch_size == batch_size`.
#[must_use]
pub fn partition_min_pages_per_batch(
    max_grid_size: u32,
    gdy: u32,
    num_pages: &[i32],
    min_num_pages_per_batch: u32,
) -> (u32, u32) {
    let mut low = min_num_pages_per_batch;
    let mut high = 0u32;
    for &elem in num_pages {
        high = cuda_max_u32_i32(high, elem);
    }
    while low < high {
        let mid = (low + high) / 2;
        let mut new_batch_size = 0u32;
        for &elem in num_pages {
            new_batch_size =
                new_batch_size.wrapping_add(ceil_div_i32_in_u32(elem, mid) as u32);
        }
        if new_batch_size.wrapping_mul(gdy) > max_grid_size {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let mut new_batch_size = 0u32;
    for &elem in num_pages {
        new_batch_size =
            new_batch_size.wrapping_add(ceil_div_i32_in_u32(elem.max(1), low) as u32);
    }
    (low, new_batch_size)
}

/// The host half of `BatchDecodeWithPagedKVCacheWorkEstimationDispatched`.
///
/// `max_grid_size` is `num_blocks_per_sm * num_sm` from the occupancy query;
/// see the module header for why it is a parameter.
///
/// The `min_num_pages_per_batch` handed to the search is `max(128 / page_size,
/// 1)` — a chunk of at least 128 tokens, so that a split does not produce work
/// items too small to amortise their own merge.
///
/// # Errors
///
/// [`Error::IndptrTooShort`] if `kv_indptr` is shorter than `batch_size + 1`.
pub fn estimate(req: &Request<'_>, max_grid_size: u32) -> Result<WorkEstimate, Error> {
    req.check()?;
    let gdy = req.num_qo_heads / req.gqa_group_size;

    if req.batch_size.wrapping_mul(gdy) >= max_grid_size {
        // The grid is already full: one work item per request, and the chunk
        // size becomes "the largest request", so nothing is split.
        let mut max_num_pages_per_batch = 1u32;
        for i in 0..req.batch_size as usize {
            max_num_pages_per_batch = cuda_max_u32_i32(max_num_pages_per_batch, req.pages(i));
        }
        return Ok(WorkEstimate {
            split_kv: false,
            max_grid_size,
            kv_chunk_size_in_pages: max_num_pages_per_batch,
            new_batch_size: req.batch_size,
            gdy,
        });
    }

    let num_pages: Vec<i32> = (0..req.batch_size as usize).map(|i| req.pages(i)).collect();
    let (max_num_pages_per_batch, new_batch_size) = partition_min_pages_per_batch(
        max_grid_size,
        gdy,
        &num_pages,
        (128 / req.page_size).max(1),
    );
    // A split that produced no more work than the batch already had is not a
    // split -- unless a graph is being captured, where the padded grid must be
    // the same shape on every replay and the partition-KV kernel is the only
    // one whose shape is fixed.
    let split_kv = !(new_batch_size == req.batch_size && !req.enable_cuda_graph);
    Ok(WorkEstimate {
        split_kv,
        max_grid_size,
        kv_chunk_size_in_pages: max_num_pages_per_batch,
        new_batch_size,
        gdy,
    })
}

/// `DecodeSplitKVIndptr` — the three index arrays, from page counts.
///
/// Returns `(request_indices, kv_tile_indices, o_indptr)`. `o_indptr` has
/// `batch_size + 1` entries and the other two have `new_batch_size`, which is
/// the shape mismatch behind the CUDA-graph overflow hazard noted in
/// [`plan`].
#[must_use]
pub fn split_kv_indptr(
    kv_indptr: &[i32],
    batch_size: u32,
    kv_chunk_size: u32,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut request_indices = Vec::new();
    let mut kv_tile_indices = Vec::new();
    let mut o_indptr = vec![0i32];

    for batch_idx in 0..batch_size as usize {
        let pages = kv_indptr[batch_idx + 1].wrapping_sub(kv_indptr[batch_idx]) as u32;
        let num_chunks_kv = ceil_div_u32(pages.max(1), kv_chunk_size);
        for kv_tile_idx in 0..num_chunks_kv {
            request_indices.push(batch_idx as i32);
            kv_tile_indices.push(kv_tile_idx as i32);
        }
        let back = *o_indptr.last().expect("o_indptr starts with a zero");
        o_indptr.push((back as u32).wrapping_add(num_chunks_kv) as i32);
    }

    (request_indices, kv_tile_indices, o_indptr)
}

/// `DecodePlan` — the plan, and the bytes to upload under it.
///
/// # The hazard worth naming
///
/// `padded_batch_size` is `new_batch_size` without graphs, and under graphs it
/// is `max_grid_size / gdy` when splitting or `batch_size` when not. The index
/// arrays are sized for `padded_batch_size` but *filled* from
/// [`split_kv_indptr`], which produces `new_batch_size` entries — and those two
/// are equal only when nobody has forced `split_kv` off after the estimator
/// ran. `attention_flashinfer_common.cuh` does exactly that, and carries a
/// twenty-line comment about the corruption it caused: work items past the
/// allocation overwrote the next array, and requests answered from each other's
/// pages. Upstream cannot detect it (a raw `std::copy` past the end);
/// [`Staging`] does, and returns [`Error::WorkspaceOverflow`].
///
/// # Errors
///
/// [`Error::WorkspaceOverflow`] if the workspace cannot hold the descriptor,
/// [`Error::IndptrTooShort`] if `kv_indptr` is too short.
pub fn plan(
    req: &Request<'_>,
    max_grid_size: u32,
    workspace: Workspace,
) -> Result<Plan<DecodePlanInfo>, Error> {
    plan_impl(req, max_grid_size, workspace, Staging::new(workspace.int_bytes))
}

/// `DecodePlanWorkspaceSize` — the same arithmetic with the writes turned off.
///
/// # Errors
///
/// [`Error::IndptrTooShort`] if `kv_indptr` is too short. Cannot overflow: the
/// sizing allocators are unbounded, which is upstream's behaviour and the
/// reason this answer is a requirement rather than a check.
pub fn workspace_size(req: &Request<'_>, max_grid_size: u32) -> Result<Sizes, Error> {
    let plan = plan_impl(req, max_grid_size, Workspace::unbounded(), Staging::sizing())?;
    Ok(Sizes { float_bytes: plan.float_bytes, int_bytes: plan.int_bytes })
}

fn plan_impl(
    req: &Request<'_>,
    max_grid_size: u32,
    workspace: Workspace,
    mut staging: Staging,
) -> Result<Plan<DecodePlanInfo>, Error> {
    let est = estimate(req, max_grid_size)?;

    let padded_batch_size: u64 = if req.enable_cuda_graph {
        if est.split_kv { u64::from(max_grid_size / est.gdy) } else { u64::from(req.batch_size) }
    } else {
        u64::from(est.new_batch_size)
    };

    let mut info = DecodePlanInfo {
        enable_cuda_graph: req.enable_cuda_graph,
        split_kv: est.split_kv,
        padded_batch_size: padded_batch_size as i64,
        ..DecodePlanInfo::default()
    };

    let (request_indices, kv_tile_indices, o_indptr) =
        split_kv_indptr(req.kv_indptr, req.batch_size, est.kv_chunk_size_in_pages);

    let padded = padded_batch_size as usize;
    let mut int_alloc = AlignedAllocator::new(workspace.int_bytes);
    info.request_indices_offset =
        int_alloc.alloc(padded * 4, 16, "batch_decode_request_indices")? as i64;
    info.kv_tile_indices_offset =
        int_alloc.alloc(padded * 4, 16, "batch_decode_kv_tile_indices")? as i64;
    info.o_indptr_offset = int_alloc.alloc((padded + 1) * 4, 16, "batch_decode_o_indptr")? as i64;
    info.kv_chunk_size_ptr_offset =
        int_alloc.alloc(4, 1, "batch_decode_kv_chunk_size_ptr")? as i64;

    if staging.materialises() {
        staging.put_i32s(
            info.request_indices_offset as usize,
            &request_indices,
            "batch_decode_request_indices",
        )?;
        staging.put_i32s(
            info.kv_tile_indices_offset as usize,
            &kv_tile_indices,
            "batch_decode_kv_tile_indices",
        )?;
        staging.put_i32s(info.o_indptr_offset as usize, &o_indptr, "batch_decode_o_indptr")?;
        // The one place pages become tokens.
        staging.put_i32(
            info.kv_chunk_size_ptr_offset as usize,
            est.kv_chunk_size_in_pages.wrapping_mul(req.page_size) as i32,
            "batch_decode_kv_chunk_size_ptr",
        )?;
    }

    // Unbounded unless the plan splits: a non-splitting plan carves no float
    // workspace at all, and reports zero rather than refusing to size one.
    let mut float_alloc = AlignedAllocator::unbounded();
    if est.split_kv {
        if staging.materialises() {
            float_alloc = AlignedAllocator::new(workspace.float_bytes);
        }
        let heads = u64::from(req.num_qo_heads);
        let head_dim = u64::from(req.head_dim);
        info.v_offset = float_alloc.alloc(
            (heads * padded_batch_size * head_dim * 4) as usize,
            16,
            "batch_decode_tmp_v",
        )? as i64;
        info.s_offset =
            float_alloc.alloc((heads * padded_batch_size * 4) as usize, 16, "batch_decode_tmp_s")?
                as i64;
        info.block_valid_mask_offset =
            int_alloc.alloc(padded, 16, "batch_decode_block_valid_mask")? as i64;
        if staging.materialises() {
            staging.put_bools(
                info.block_valid_mask_offset as usize,
                (0..padded).map(|i| (i as u32) < est.new_batch_size),
                "batch_decode_block_valid_mask",
            )?;
        }
    }

    let int_bytes = int_alloc.used();
    Ok(Plan {
        info,
        int_upload: staging.into_upload(int_bytes),
        int_bytes,
        float_bytes: float_alloc.used(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(kv_indptr: &'a [i32], batch_size: u32) -> Request<'a> {
        Request {
            kv_indptr,
            batch_size,
            num_qo_heads: 32,
            gqa_group_size: 4,
            page_size: 16,
            head_dim: 128,
            enable_cuda_graph: false,
        }
    }

    /// A batch that already fills the grid is never split, and the chunk size
    /// becomes the largest request so that every request is one work item.
    #[test]
    fn a_full_grid_is_not_split() {
        let indptr: Vec<i32> = (0..=64).map(|i| i * 10).collect();
        let est = estimate(&request(&indptr, 64), 16).unwrap();
        assert!(!est.split_kv);
        assert_eq!(est.new_batch_size, 64);
        assert_eq!(est.kv_chunk_size_in_pages, 10);
        assert_eq!(est.gdy, 8);
    }

    /// One request on a big machine is split until the grid is full and no
    /// further: 4096 pages over a 1024-CTA grid with 8 KV heads is 128 chunks,
    /// so the search stops at 32 pages -- *not* at the 8-page floor, which is
    /// only a lower bound on where the search starts.
    #[test]
    fn a_lone_request_is_split_until_the_grid_is_full() {
        let indptr = [0i32, 4096];
        let est = estimate(&request(&indptr, 1), 1024).unwrap();
        assert!(est.split_kv);
        assert_eq!(est.kv_chunk_size_in_pages, 32);
        assert_eq!(est.new_batch_size, 128);
    }

    /// The empty batch: no work items, no refusal, and a chunk size of the
    /// floor. Decode is the one planner that survives it.
    #[test]
    fn the_empty_batch_plans() {
        let indptr = [0i32];
        let plan = plan(&request(&indptr, 0), 1024, Workspace::new(1 << 20, 1 << 20)).unwrap();
        assert_eq!(plan.info.padded_batch_size, 0);
        assert!(!plan.info.split_kv);
    }

    /// A workspace too small to hold the descriptor is refused, by name.
    #[test]
    fn a_workspace_that_cannot_hold_the_plan_is_refused() {
        let indptr: Vec<i32> = (0..=64).map(|i| i * 10).collect();
        let err = plan(&request(&indptr, 64), 16, Workspace::new(1 << 20, 8)).unwrap_err();
        assert!(matches!(err, Error::WorkspaceOverflow { .. }));
    }

    /// The sizing pass answers what the materialising pass consumes.
    #[test]
    fn sizing_agrees_with_planning() {
        let indptr = [0i32, 4096];
        let req = request(&indptr, 1);
        let sizes = workspace_size(&req, 1024).unwrap();
        let plan = plan(&req, 1024, Workspace::new(1 << 24, 1 << 20)).unwrap();
        assert_eq!(sizes.int_bytes, plan.int_bytes);
        assert_eq!(sizes.float_bytes, plan.float_bytes);
    }
}
