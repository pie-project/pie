//! The fa2 decode work split: FlashInfer's host planner, transcribed. Given
//! the host copy of the kv page indptr it decides whether to split kv,
//! partitions requests into `(request, kv tile)` work items, and stages the
//! index vectors the decode kernel walks.

use new_kernels::KernelError;

use super::alloc::{AlignedAllocator, Staging};
use super::arith::{ceil_div_i32_in_u32, ceil_div_u32, cuda_max_u32_i32};
use super::info::DecodePlanInfo;
use super::{Built, Sizes, refuse_len};

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Host copy of the kv page indptr — `[batch_size + 1]`.
    pub kv_indptr: &'a [i32],
    pub batch_size: u32,
    pub num_qo_heads: u32,
    pub gqa_group_size: u32,
    pub page_size: u32,
    pub head_dim: u32,
    pub enable_cuda_graph: bool,
}

impl Request<'_> {
    fn pages(&self, i: usize) -> i32 {
        self.kv_indptr[i + 1].wrapping_sub(self.kv_indptr[i])
    }

    fn check(&self, op: &'static str) -> Result<(), KernelError> {
        refuse_len(op, "kv_indptr", self.kv_indptr.len(), self.batch_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkEstimate {
    pub split_kv: bool,
    pub max_grid_size: u32,
    pub kv_chunk_size_in_pages: u32,
    pub new_batch_size: u32,
    pub gdy: u32,
}

pub fn estimate(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
) -> Result<WorkEstimate, KernelError> {
    #[must_use]
    fn partition_min_pages_per_batch(
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
                new_batch_size = new_batch_size.wrapping_add(ceil_div_i32_in_u32(elem, mid) as u32);
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

    req.check(op)?;
    let gdy = req.num_qo_heads / req.gqa_group_size;

    if req.batch_size.wrapping_mul(gdy) >= max_grid_size {
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
    let (max_num_pages_per_batch, new_batch_size) =
        partition_min_pages_per_batch(max_grid_size, gdy, &num_pages, (128 / req.page_size).max(1));
    let split_kv = !(new_batch_size == req.batch_size && !req.enable_cuda_graph);
    Ok(WorkEstimate {
        split_kv,
        max_grid_size,
        kv_chunk_size_in_pages: max_num_pages_per_batch,
        new_batch_size,
        gdy,
    })
}

pub fn plan(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
    int_bytes: usize,
    float_bytes: usize,
) -> Result<Built<DecodePlanInfo>, KernelError> {
    plan_impl(
        op,
        req,
        max_grid_size,
        int_bytes,
        float_bytes,
        Staging::new(op, int_bytes),
    )
}

/// The sizing pass: the same body over an unbounded allocator and an empty
/// staging buffer, so the driver can grant a workspace before granting one.
pub fn workspace_size(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
) -> Result<Sizes, KernelError> {
    let built = plan_impl(
        op,
        req,
        max_grid_size,
        usize::MAX,
        usize::MAX,
        Staging::sizing(op),
    )?;
    Ok(Sizes {
        float_bytes: built.float_bytes,
        int_bytes: built.int_bytes,
    })
}

fn plan_impl(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
    int_bytes: usize,
    float_bytes: usize,
    mut staging: Staging,
) -> Result<Built<DecodePlanInfo>, KernelError> {
    #[must_use]
    fn split_kv_indptr(
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

    let est = estimate(op, req, max_grid_size)?;

    let padded_batch_size: u64 = if req.enable_cuda_graph {
        if est.split_kv {
            u64::from(max_grid_size / est.gdy)
        } else {
            u64::from(req.batch_size)
        }
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
    let mut int_alloc = AlignedAllocator::new(op, int_bytes);
    info.request_indices_offset =
        int_alloc.alloc(padded * 4, 16, "batch_decode_request_indices")? as i64;
    info.kv_tile_indices_offset =
        int_alloc.alloc(padded * 4, 16, "batch_decode_kv_tile_indices")? as i64;
    info.o_indptr_offset = int_alloc.alloc((padded + 1) * 4, 16, "batch_decode_o_indptr")? as i64;
    info.kv_chunk_size_ptr_offset = int_alloc.alloc(4, 1, "batch_decode_kv_chunk_size_ptr")? as i64;

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
        staging.put_i32s(
            info.o_indptr_offset as usize,
            &o_indptr,
            "batch_decode_o_indptr",
        )?;
        staging.put_i32(
            info.kv_chunk_size_ptr_offset as usize,
            est.kv_chunk_size_in_pages.wrapping_mul(req.page_size) as i32,
            "batch_decode_kv_chunk_size_ptr",
        )?;
    }

    let mut float_alloc = AlignedAllocator::unbounded(op);
    if est.split_kv {
        if staging.materialises() {
            float_alloc = AlignedAllocator::new(op, float_bytes);
        }
        let heads = u64::from(req.num_qo_heads);
        let head_dim = u64::from(req.head_dim);
        info.v_offset = float_alloc.alloc(
            (heads * padded_batch_size * head_dim * 4) as usize,
            16,
            "batch_decode_tmp_v",
        )? as i64;
        info.s_offset = float_alloc.alloc(
            (heads * padded_batch_size * 4) as usize,
            16,
            "batch_decode_tmp_s",
        )? as i64;
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
    Ok(Built {
        info,
        int_upload: staging.into_upload(int_bytes),
        int_bytes,
        float_bytes: float_alloc.used(),
    })
}
