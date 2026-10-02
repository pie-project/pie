//! The sparse-attention indexer (DeepSeek DSA, Qwen 3.8 QSA) and the pooled
//! block boundaries it reads: a small per-token key cached in its own paged
//! pool; every `ratio`-th position closes a block whose mean key is filed at
//! the closing cell; per query, a ReLU-gated multi-head score against the
//! closed blocks' keys and the `top_k` by a bisected threshold. Reference:
//! kernels-xla `attn::index`, `attn::pool`.
//!
//! These are table walks and a few dots over a vocabulary-sized pool the
//! host already holds, so the host runs them; the selected readers are the
//! paged attention with a block selection (`paged::decode_selected`).

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::program::HostOp;
use crate::tensor::{KvPool, RaggedTensor, Tensor};

fn paged(op: &'static str, pool: &KvPool) -> Result<u32, Error> {
    expect(op, pool.keys, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, pool.page_indices, &[Dtype::I32])?;
    expect(op, pool.page_indptr, &[Dtype::I32])?;
    if pool.page_size <= 0 {
        return Err(refuse(op, "the pool's page size is zero"));
    }
    Ok(pool.page_size as u32)
}

/// Whether one PE holds these buffers' words (each within an array, all
/// within the data budget); otherwise the op runs on the host.
fn one_pe(words: &[u64]) -> bool {
    use crate::linear::gemm::{ARRAY_WORDS, pe_words};
    words.iter().all(|w| *w <= ARRAY_WORDS) && words.iter().sum::<u64>() + 64 <= pe_words()
}

fn long_enough(op: &'static str, what: &str, t: Tensor, n: u32) -> Result<(), Error> {
    if t.elements() < u64::from(n) {
        return Err(refuse(
            op,
            format!(
                "the {what} holds {} entries and this op reads {n}",
                t.elements()
            ),
        ));
    }
    Ok(())
}

/// Lands row `i` of the index key `k` in pool row `write_page[i] ·
/// page_size + write_offset[i]`; a page or offset out of range drops the row.
pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    keys: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_kv_append";
    expect(OP, k, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, write_page, &[Dtype::I32])?;
    expect(OP, write_offset, &[Dtype::I32])?;
    let ps = paged(OP, keys)?;
    if keys.keys.width != k.width || k.rows == 0 {
        return Err(refuse(
            OP,
            format!(
                "the pool's row is {} wide, not the {}-wide row this index writes",
                keys.keys.width, k.width
            ),
        ));
    }
    long_enough(OP, "write page table", write_page, k.rows)?;
    long_enough(OP, "write offset table", write_offset, k.rows)?;
    let (rows, width, table_rows) = (k.rows, k.width, keys.keys.rows);
    let on_pe = one_pe(&[
        k.elements(),
        keys.keys.elements(),
        write_page.elements(),
        write_offset.elements(),
    ]);
    ctx.emit(&mut |cx| {
        let kb = cx.read(k)?;
        let pb = cx.read(write_page)?;
        let ob = cx.read(write_offset)?;
        cx.read(keys.keys)?;
        let tb = cx.write(keys.keys)?;
        if !on_pe {
            cx.host(HostOp::PageWrite {
                src: kb.name,
                table: tb.name,
                write_page: pb.name,
                write_offset: ob.name,
                rows,
                width,
                page_size: ps,
                table_rows,
            });
            return Ok(());
        }
        cx.library("k_index_page_write");
        cx.call(
            "k_index_page_write",
            vec![
                Arg::Ptr(kb),
                Arg::Ptr(tb),
                Arg::Ptr(pb),
                Arg::Ptr(ob),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(width)),
                Arg::Int(i64::from(ps)),
                Arg::Int(i64::from(table_rows)),
            ],
        );
        Ok(())
    })
}

fn boundary(
    ctx: &Ctx<'_>,
    op: &'static str,
    positions: Tensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    let rows = boundary_pos.rows;
    if rows == 0 || ratio == 0 {
        return Err(refuse(op, "the rows and the pooling ratio are nonzero"));
    }
    for t in [
        positions,
        request_of_token,
        row_valid,
        boundary_pos,
        boundary_req,
        boundary_rope,
    ] {
        expect(op, t, &[Dtype::I32])?;
    }
    if boundary_req.rows != rows || boundary_rope.rows != rows {
        return Err(refuse(
            op,
            "the boundary tables are one entry per token row",
        ));
    }
    long_enough(op, "position table", positions, rows)?;
    long_enough(op, "owning-request table", request_of_token, rows)?;
    long_enough(op, "row-valid table", row_valid, rows)?;
    // One PE marks every row (the tables are small: a few words a row); a
    // fire whose tables outgrow a PE marks on the host.
    let words = positions.elements()
        + request_of_token.elements()
        + row_valid.elements()
        + 3 * u64::from(rows);
    let on_pe = words <= crate::linear::gemm::pe_words();
    ctx.emit(&mut |cx| {
        let p = cx.read(positions)?;
        let r = cx.read(request_of_token)?;
        let v = cx.read(row_valid)?;
        let bp = cx.write(boundary_pos)?;
        let br = cx.write(boundary_req)?;
        let bo = cx.write(boundary_rope)?;
        if !on_pe {
            cx.host(HostOp::Boundary {
                positions: p.name,
                request_of_token: r.name,
                row_valid: v.name,
                boundary_pos: bp.name,
                boundary_req: br.name,
                boundary_rope: bo.name,
                rows,
                ratio,
            });
            return Ok(());
        }
        cx.library("k_boundary");
        cx.call(
            "k_boundary",
            vec![
                Arg::Ptr(p),
                Arg::Ptr(r),
                Arg::Ptr(v),
                Arg::Ptr(bp),
                Arg::Ptr(br),
                Arg::Ptr(bo),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(ratio)),
            ],
        );
        Ok(())
    })
}

/// Marks the rows whose position closes a `ratio` block: `boundary_pos` the
/// position (or -1), `boundary_rope` the block's first position, and
/// `boundary_req` each row's request in the fire.
pub fn boundary_decode(
    ctx: &Ctx<'_>,
    positions: Tensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    boundary(
        ctx,
        "attention.pool_boundary_decode",
        positions,
        request_of_token,
        row_valid,
        ratio,
        boundary_pos,
        boundary_req,
        boundary_rope,
    )
}

pub fn boundary_prefill(
    ctx: &Ctx<'_>,
    positions: RaggedTensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    boundary(
        ctx,
        "attention.pool_boundary_prefill",
        positions.data,
        request_of_token,
        row_valid,
        ratio,
        boundary_pos,
        boundary_req,
        boundary_rope,
    )
}

/// Each boundary row's mean of the `ratio` cached keys ending at its
/// position (positions before 0 add nothing; the divisor stays `ratio`);
/// 0 for a row closing no block.
pub fn block_mean(
    ctx: &Ctx<'_>,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    keys: &KvPool,
    head_dim: u32,
    ratio: u32,
    entries: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_block_mean";
    expect(OP, entries, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, boundary_pos, &[Dtype::I32])?;
    expect(OP, boundary_req, &[Dtype::I32])?;
    let ps = paged(OP, keys)?;
    if head_dim == 0 || ratio == 0 {
        return Err(refuse(OP, "the key width and the block width are nonzero"));
    }
    if entries.width != head_dim || keys.keys.width != head_dim {
        return Err(refuse(
            OP,
            "the stated head width is not the entry's and the pool's width",
        ));
    }
    let rows = entries.rows;
    if rows == 0 || boundary_pos.rows != rows {
        return Err(refuse(
            OP,
            "the boundary tables and entries are one row per token row",
        ));
    }
    long_enough(OP, "boundary request table", boundary_req, rows)?;
    let on_pe = one_pe(&[
        boundary_pos.elements(),
        boundary_req.elements(),
        keys.keys.elements(),
        keys.page_indices.elements(),
        keys.page_indptr.elements(),
        entries.elements(),
    ]);
    ctx.emit(&mut |cx| {
        let bp = cx.read(boundary_pos)?;
        let br = cx.read(boundary_req)?;
        let kb = cx.read(keys.keys)?;
        let ib = cx.read(keys.page_indices)?;
        let pb = cx.read(keys.page_indptr)?;
        let eb = cx.write(entries)?;
        if !on_pe {
            cx.host(HostOp::BlockMean {
                boundary_pos: bp.name,
                boundary_req: br.name,
                keys: kb.name,
                indices: ib.name,
                indptr: pb.name,
                entries: eb.name,
                rows,
                head_dim,
                ratio,
                page_size: ps,
            });
            return Ok(());
        }
        cx.library("k_index_cell");
        cx.library("k_index_block_mean");
        cx.call(
            "k_index_block_mean",
            vec![
                Arg::Ptr(bp),
                Arg::Ptr(br),
                Arg::Ptr(kb),
                Arg::Ptr(ib.clone()),
                Arg::Ptr(pb.clone()),
                Arg::Int(ib.len() as i64),
                Arg::Int(pb.len() as i64),
                Arg::Ptr(eb),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(head_dim)),
                Arg::Int(i64::from(ratio)),
                Arg::Int(i64::from(ps)),
                Arg::Int(i64::from(keys.keys.rows)),
            ],
        );
        Ok(())
    })
}

/// Files each boundary row's entry at its pool cell (its request's page
/// table at its position); a row closing no block writes nothing.
pub fn pool_kv_append(
    ctx: &Ctx<'_>,
    entries: Tensor,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_kv_append";
    let _ = (write_page, write_offset);
    expect(OP, entries, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, boundary_pos, &[Dtype::I32])?;
    expect(OP, boundary_req, &[Dtype::I32])?;
    let ps = paged(OP, pool)?;
    let rows = entries.rows;
    if rows == 0 || pool.keys.width != entries.width {
        return Err(refuse(
            OP,
            "the compressed pool's row is the entry's width, one row per token row",
        ));
    }
    long_enough(OP, "boundary position table", boundary_pos, rows)?;
    long_enough(OP, "boundary request table", boundary_req, rows)?;
    let width = entries.width;
    let on_pe = one_pe(&[
        entries.elements(),
        boundary_pos.elements(),
        boundary_req.elements(),
        pool.keys.elements(),
        pool.page_indices.elements(),
        pool.page_indptr.elements(),
    ]);
    ctx.emit(&mut |cx| {
        let eb = cx.read(entries)?;
        let bp = cx.read(boundary_pos)?;
        let br = cx.read(boundary_req)?;
        let ib = cx.read(pool.page_indices)?;
        let pb = cx.read(pool.page_indptr)?;
        cx.read(pool.keys)?;
        let kb = cx.write(pool.keys)?;
        if !on_pe {
            cx.host(HostOp::PoolWrite {
                entries: eb.name,
                boundary_pos: bp.name,
                boundary_req: br.name,
                keys: kb.name,
                indices: ib.name,
                indptr: pb.name,
                rows,
                width,
                page_size: ps,
            });
            return Ok(());
        }
        cx.library("k_index_cell");
        cx.library("k_index_pool_write");
        cx.call(
            "k_index_pool_write",
            vec![
                Arg::Ptr(eb),
                Arg::Ptr(bp),
                Arg::Ptr(br),
                Arg::Ptr(kb),
                Arg::Ptr(ib.clone()),
                Arg::Ptr(pb.clone()),
                Arg::Int(ib.len() as i64),
                Arg::Int(pb.len() as i64),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(width)),
                Arg::Int(i64::from(ps)),
                Arg::Int(i64::from(pool.keys.rows)),
            ],
        );
        Ok(())
    })
}

/// Per query row: scores `Σ_h max(q_h · k_j, 0) · w_h` against the cached
/// keys `j < (pos + 1) / ratio` (key `j` at position `(j + 1) · ratio − 1`),
/// then the `top_k` keys: every key when there are no more than `top_k`,
/// else, in key order, those scoring at or above a threshold bisected 40
/// times between the row's min and max (as the GPU kernels pick them);
/// unfilled slots are -1.
pub fn topk(
    ctx: &Ctx<'_>,
    q: Tensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    selection: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_topk";
    expect(OP, q, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, selection, &[Dtype::I32])?;
    expect(OP, positions, &[Dtype::I32])?;
    expect(OP, request_of_token, &[Dtype::I32])?;
    if let Some(w) = weights {
        expect(OP, w, &[Dtype::F32, Dtype::Bf16])?;
    }
    let ps = paged(OP, keys)?;
    if heads == 0 || head_dim == 0 || top_k == 0 {
        return Err(refuse(
            OP,
            "the head count, key width and selection budget are nonzero",
        ));
    }
    if q.width != heads * head_dim || keys.keys.width != head_dim {
        return Err(refuse(OP, "the index query is not heads x key width"));
    }
    if weights.is_some_and(|w| w.width != heads) {
        return Err(refuse(
            OP,
            "the index head weights are not one per stated head",
        ));
    }
    if selection.width != top_k {
        return Err(refuse(OP, "the selection is not the budget it states"));
    }
    let rows = selection.rows;
    if rows == 0 || q.rows < rows {
        return Err(refuse(OP, "q is shorter than the selection"));
    }
    long_enough(OP, "position table", positions, rows)?;
    long_enough(OP, "owning-request table", request_of_token, rows)?;
    if u64::from(keys.max_pages) * u64::from(ps) / u64::from(ratio.max(1)) == 0 {
        return Err(refuse(OP, "the pool's page bound holds no key"));
    }
    let nk = u64::from(keys.max_pages) * u64::from(ps) / u64::from(ratio.max(1));
    let on_pe = one_pe(&[
        q.elements(),
        weights.map_or(0, |w| w.elements()),
        keys.keys.elements(),
        keys.page_indices.elements(),
        keys.page_indptr.elements(),
        positions.elements(),
        request_of_token.elements(),
        selection.elements(),
        nk,
    ]);
    ctx.emit(&mut |cx| {
        let qb = cx.read(q)?;
        let wb = match weights {
            Some(w) => Some(cx.read(w)?),
            None => None,
        };
        let kb = cx.read(keys.keys)?;
        let ib = cx.read(keys.page_indices)?;
        let pb = cx.read(keys.page_indptr)?;
        let pos = cx.read(positions)?;
        let req = cx.read(request_of_token)?;
        let sb = cx.write(selection)?;
        if !on_pe {
            cx.host(HostOp::IndexTopk {
                q: qb.name,
                weights: wb.map(|b| b.name),
                keys: kb.name,
                indices: ib.name,
                indptr: pb.name,
                positions: pos.name,
                request_of_token: req.name,
                selection: sb.name,
                rows,
                heads,
                head_dim,
                top_k,
                ratio: ratio.max(1),
                page_size: ps,
                max_pages: keys.max_pages,
            });
            return Ok(());
        }
        let scores = cx.scratch("scores", nk.max(1));
        let (w_ptr, has_w) = match &wb {
            Some(w) => (Arg::Ptr(w.clone()), true),
            None => (Arg::Dummy("f32"), false),
        };
        cx.library("k_index_cell");
        cx.library("k_index_topk");
        cx.call(
            "k_index_topk",
            vec![
                Arg::Ptr(qb),
                w_ptr,
                Arg::Bool(has_w),
                Arg::Ptr(kb),
                Arg::Ptr(ib.clone()),
                Arg::Ptr(pb.clone()),
                Arg::Int(ib.len() as i64),
                Arg::Int(pb.len() as i64),
                Arg::Ptr(pos),
                Arg::Ptr(req),
                Arg::Ptr(sb),
                Arg::Scratch(scores.clone(), "f32"),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(heads)),
                Arg::Int(i64::from(head_dim)),
                Arg::Int(i64::from(top_k)),
                Arg::Int(i64::from(ratio.max(1))),
                Arg::Int(i64::from(ps)),
                Arg::Int(nk as i64),
                Arg::Int(i64::from(keys.keys.rows)),
            ],
        );
        Ok(())
    })
}
