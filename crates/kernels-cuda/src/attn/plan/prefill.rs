use super::alloc::{AlignedAllocator, Staging};
use super::arith::{ceil_div_i64, ceil_div_u32, fa2_determine_cta_tile_q};
use super::info::PrefillPlanInfo;
use super::{Device, Error, Plan, Sizes, Workspace};

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub qo_indptr: &'a [i32],
    pub kv_indptr: &'a [i32],
    pub total_num_rows: u32,
    pub batch_size: u32,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim_qk: u32,
    pub head_dim_vo: u32,
    pub page_size: u32,
    pub enable_cuda_graph: bool,
    pub sizeof_dtype_o: u32,
    pub window_left: i32,
    pub fixed_split_size: i32,
    pub disable_split_kv: bool,
    pub num_colocated_ctas: i64,
}

impl Request<'_> {
    fn check(&self) -> Result<(), Error> {
        let needed = self.batch_size as usize + 1;
        if self.qo_indptr.len() < needed {
            return Err(Error::IndptrTooShort {
                array: "qo_indptr",
                needed,
                got: self.qo_indptr.len(),
            });
        }
        if self.kv_indptr.len() < needed {
            return Err(Error::IndptrTooShort {
                array: "kv_indptr",
                needed,
                got: self.kv_indptr.len(),
            });
        }
        if self.num_kv_heads == 0 || !self.num_qo_heads.is_multiple_of(self.num_kv_heads) {
            return Err(Error::HeadsNotDivisible {
                num_qo_heads: self.num_qo_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub split_kv: bool,
    pub new_batch_size: u32,
    pub padded_batch_size: u64,
    pub cta_tile_q: u32,
    pub kv_chunk_size: i64,
    pub request_indices: Vec<i32>,
    pub qo_tile_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    pub merge_indptr: Vec<i32>,
    pub o_indptr: Vec<i32>,
}

pub fn split_qo_kv_indptr(
    req: &Request<'_>,
    max_batch_size_if_split: u32,
    cc_major: i32,
) -> Result<Split, Error> {
    #[must_use]
    pub fn binary_search_kv_chunk_size(
        enable_cuda_graph: bool,
        max_batch_size_if_split: u32,
        packed_qo_len_arr: &[i64],
        kv_len_arr: &[i64],
        qo_chunk_size: u32,
        min_kv_chunk_size: i64,
    ) -> (bool, i64) {
        let batch_size = packed_qo_len_arr.len();
        let mut max_kv_len: i64 = 1;
        for &kv_len in kv_len_arr {
            max_kv_len = max_kv_len.max(kv_len);
        }

        let mut low = min_kv_chunk_size;
        let mut high = max_kv_len;
        const MIN_KV_LEN: i64 = 1;
        while low < high {
            let mid = (low + high) / 2;
            let mut new_batch_size: i64 = 0;
            for i in 0..batch_size {
                new_batch_size = new_batch_size.wrapping_add(
                    ceil_div_i64(packed_qo_len_arr[i], i64::from(qo_chunk_size))
                        .wrapping_mul(ceil_div_i64(kv_len_arr[i].max(MIN_KV_LEN), mid)),
                );
            }
            if new_batch_size > i64::from(max_batch_size_if_split) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        (enable_cuda_graph || low < max_kv_len, low)
    }

    let batch_size = req.batch_size as usize;
    let gqa_group_size = req.num_qo_heads / req.num_kv_heads;

    let mut packed_qo_len_arr = vec![0i64; batch_size];
    let mut kv_len_arr = vec![0i64; batch_size];
    for i in 0..batch_size {
        packed_qo_len_arr[i] = i64::from(req.qo_indptr[i + 1].wrapping_sub(req.qo_indptr[i]))
            * i64::from(gqa_group_size);
        if packed_qo_len_arr[i] < 0 {
            return Err(Error::NegativeSpan {
                array: "qo_indptr",
                index: i,
                begin: i64::from(req.qo_indptr[i]),
                end: i64::from(req.qo_indptr[i + 1]),
            });
        }
        kv_len_arr[i] = i64::from(req.kv_indptr[i + 1].wrapping_sub(req.kv_indptr[i]));
        if kv_len_arr[i] < 0 {
            return Err(Error::NegativeSpan {
                array: "kv_indptr",
                index: i,
                begin: i64::from(req.kv_indptr[i]),
                end: i64::from(req.kv_indptr[i + 1]),
            });
        }
    }

    let min_kv_chunk_size = (128 / req.page_size).max(1);
    let cta_tile_q: u32;
    let total_num_tiles_q: u32;
    if req.enable_cuda_graph {
        let max_seq_len = req
            .total_num_rows
            .wrapping_sub(req.batch_size)
            .wrapping_add(1);
        let max_qo_len = u64::from(max_seq_len) * u64::from(gqa_group_size);
        cta_tile_q = fa2_determine_cta_tile_q(max_qo_len as i64, req.head_dim_vo, cc_major);
        total_num_tiles_q =
            ceil_div_u32(req.total_num_rows.wrapping_mul(gqa_group_size), cta_tile_q)
                .wrapping_add(req.batch_size)
                .wrapping_sub(1);
    } else {
        if batch_size == 0 {
            return Err(Error::EmptyBatch);
        }
        let sum_packed_qo_len: i64 = packed_qo_len_arr
            .iter()
            .fold(0i64, |a, b| a.wrapping_add(*b));
        let avg_packed_qo_len = sum_packed_qo_len / i64::from(req.batch_size);
        cta_tile_q = fa2_determine_cta_tile_q(avg_packed_qo_len, req.head_dim_vo, cc_major);
        total_num_tiles_q = packed_qo_len_arr.iter().fold(0u32, |acc, &len| {
            acc.wrapping_add(ceil_div_i64(len, i64::from(cta_tile_q)) as u32)
        });
    }

    let effective_kv_len_arr: Vec<i64> = (0..batch_size)
        .map(|i| {
            let windowed = if req.window_left >= 0 {
                i64::from(ceil_div_u32(
                    (req.window_left as u32).wrapping_add(cta_tile_q),
                    req.page_size,
                ))
            } else {
                kv_len_arr[i]
            };
            windowed.min(kv_len_arr[i])
        })
        .collect();

    let mut split_kv = false;
    let mut kv_chunk_size: i64;
    if req.disable_split_kv {
        kv_chunk_size = i64::MAX;
    } else if req.fixed_split_size > 0 {
        kv_chunk_size = i64::from(req.fixed_split_size);
    } else {
        let (found_split, chunk) = binary_search_kv_chunk_size(
            req.enable_cuda_graph,
            max_batch_size_if_split,
            &packed_qo_len_arr,
            &effective_kv_len_arr,
            cta_tile_q,
            i64::from(min_kv_chunk_size),
        );
        split_kv = found_split;
        kv_chunk_size = chunk;
    }

    let mut request_indices = Vec::new();
    let mut qo_tile_indices = Vec::new();
    let mut kv_tile_indices = Vec::new();
    let mut merge_indptr = vec![0i32];
    let mut o_indptr = vec![0i32];
    let mut new_batch_size = 0u32;
    for request_idx in 0..batch_size {
        let packed_qo_len = packed_qo_len_arr[request_idx];
        let num_tiles_q = ceil_div_i64(packed_qo_len, i64::from(cta_tile_q));
        let kv_len = i64::from((effective_kv_len_arr[request_idx] as i32).max(1));
        let num_chunks_kv = if req.disable_split_kv {
            1
        } else {
            ceil_div_i64(kv_len, kv_chunk_size)
        };
        if req.fixed_split_size > 0 && !req.disable_split_kv {
            split_kv = split_kv || num_chunks_kv > 1;
        }
        for q_tile_idx in 0..num_tiles_q {
            for kv_tile_idx in 0..num_chunks_kv {
                new_batch_size = new_batch_size.wrapping_add(1);
                request_indices.push(request_idx as i32);
                qo_tile_indices.push(q_tile_idx as i32);
                kv_tile_indices.push(kv_tile_idx as i32);
            }
        }

        let qo_len = packed_qo_len / i64::from(gqa_group_size);
        for _ in 0..qo_len {
            let back = *merge_indptr
                .last()
                .expect("merge_indptr starts with a zero");
            merge_indptr.push(i64::from(back).wrapping_add(num_chunks_kv) as i32);
        }
        let back = *o_indptr.last().expect("o_indptr starts with a zero");
        o_indptr.push(i64::from(back).wrapping_add(qo_len.wrapping_mul(num_chunks_kv)) as i32);
    }

    let padded_batch_size: u64 = if req.enable_cuda_graph {
        u64::from(max_batch_size_if_split.max(total_num_tiles_q))
    } else {
        u64::from(new_batch_size)
    };
    if u64::from(new_batch_size) > padded_batch_size {
        return Err(Error::BatchExceedsPadded {
            new_batch_size: u64::from(new_batch_size),
            padded_batch_size,
        });
    }

    kv_chunk_size = kv_chunk_size.wrapping_mul(i64::from(req.page_size));

    Ok(Split {
        split_kv,
        new_batch_size,
        padded_batch_size,
        cta_tile_q,
        kv_chunk_size,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        merge_indptr,
        o_indptr,
    })
}

pub fn plan(
    req: &Request<'_>,
    device: &Device,
    workspace: Workspace,
) -> Result<Plan<PrefillPlanInfo>, Error> {
    plan_impl(req, device, workspace, Staging::new(workspace.int_bytes))
}

pub fn workspace_size(req: &Request<'_>, device: &Device) -> Result<Sizes, Error> {
    let plan = plan_impl(req, device, Workspace::unbounded(), Staging::sizing())?;
    Ok(Sizes {
        float_bytes: plan.float_bytes,
        int_bytes: plan.int_bytes,
    })
}

fn plan_impl(
    req: &Request<'_>,
    device: &Device,
    workspace: Workspace,
    mut staging: Staging,
) -> Result<Plan<PrefillPlanInfo>, Error> {
    req.check()?;

    let num_blocks_per_sm: i64 = 2;
    let available_ctas = num_blocks_per_sm * i64::from(device.num_sm) - req.num_colocated_ctas;
    let max_grid_size = available_ctas.max(0) as i32;
    let max_batch_size_if_split = (max_grid_size as u32) / req.num_kv_heads;

    let split = split_qo_kv_indptr(req, max_batch_size_if_split, device.cc_major)?;

    let mut info = PrefillPlanInfo {
        cta_tile_q: i64::from(split.cta_tile_q),
        total_num_rows: i64::from(req.total_num_rows),
        enable_cuda_graph: req.enable_cuda_graph,
        padded_batch_size: split.padded_batch_size as i64,
        split_kv: split.split_kv,
        ..PrefillPlanInfo::default()
    };

    let padded = split.padded_batch_size as usize;
    let mut int_alloc = AlignedAllocator::new(workspace.int_bytes);
    info.request_indices_offset =
        int_alloc.alloc(4 * padded, 16, "batch_prefill_request_indices")? as i64;
    info.qo_tile_indices_offset =
        int_alloc.alloc(4 * padded, 16, "batch_prefill_qo_tile_indices")? as i64;
    info.kv_tile_indices_offset =
        int_alloc.alloc(4 * padded, 16, "batch_prefill_kv_tile_indices")? as i64;
    info.o_indptr_offset = int_alloc.alloc(
        4 * (req.batch_size as usize + 1),
        16,
        "batch_prefill_o_indptr",
    )? as i64;
    info.kv_chunk_size_ptr_offset =
        int_alloc.alloc(4, 1, "batch_prefill_kv_chunk_size_ptr")? as i64;

    if info.enable_cuda_graph {
        info.total_num_rows_offset = int_alloc.alloc(4, 16, "batch_prefill_total_num_rows")? as i64;
        if staging.materialises() {
            staging.put_u32(
                info.total_num_rows_offset as usize,
                req.qo_indptr[req.batch_size as usize] as u32,
                "batch_prefill_total_num_rows",
            )?;
        }
    }

    if staging.materialises() {
        staging.put_i32s(
            info.request_indices_offset as usize,
            &split.request_indices,
            "batch_prefill_request_indices",
        )?;
        staging.put_i32s(
            info.qo_tile_indices_offset as usize,
            &split.qo_tile_indices,
            "batch_prefill_qo_tile_indices",
        )?;
        staging.put_i32s(
            info.kv_tile_indices_offset as usize,
            &split.kv_tile_indices,
            "batch_prefill_kv_tile_indices",
        )?;
        staging.put_i32s(
            info.o_indptr_offset as usize,
            &split.o_indptr,
            "batch_prefill_o_indptr",
        )?;
        staging.put_i32(
            info.kv_chunk_size_ptr_offset as usize,
            split.kv_chunk_size as i32,
            "batch_prefill_kv_chunk_size_ptr",
        )?;
    }

    let mut float_alloc = AlignedAllocator::unbounded();
    if split.split_kv {
        if staging.materialises() {
            float_alloc = AlignedAllocator::new(workspace.float_bytes);
        }
        let heads = u64::from(req.num_qo_heads);
        let tile_q = u64::from(split.cta_tile_q);
        let head_dim = u64::from(req.head_dim_vo);
        info.v_offset = float_alloc.alloc(
            (heads * split.padded_batch_size * tile_q * head_dim * 4) as usize,
            16,
            "batch_prefill_tmp_v",
        )? as i64;
        info.s_offset = float_alloc.alloc(
            (heads * split.padded_batch_size * tile_q * 4) as usize,
            16,
            "batch_prefill_tmp_s",
        )? as i64;
        info.merge_indptr_offset = int_alloc.alloc(
            4 * (info.total_num_rows as usize + 1),
            16,
            "batch_prefill_merge_indptr",
        )? as i64;
        info.block_valid_mask_offset =
            int_alloc.alloc(padded, 16, "batch_prefill_block_valid_mask")? as i64;

        if staging.materialises() {
            staging.put_i32s(
                info.merge_indptr_offset as usize,
                &split.merge_indptr,
                "batch_prefill_merge_indptr",
            )?;
            staging.put_bools(
                info.block_valid_mask_offset as usize,
                (0..padded).map(|i| (i as u32) < split.new_batch_size),
                "batch_prefill_block_valid_mask",
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
