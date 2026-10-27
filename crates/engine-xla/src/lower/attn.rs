//! The attention kernels lowered from handles. A paged, MLA or pooled
//! reader is a call a kernel builds from handles (its shapes checked); the
//! call reads its planes in the order the kernel consumes them. The other
//! kernels read their operands here, in that order too. Every result lands
//! in the handle it updates.
#![allow(clippy::too_many_arguments)]

use kernels_xla::attn::paged;

use crate::emit::Ctx;
use crate::handle::Paged;
use crate::handle::{KvPool, Tensor};
use kernels_xla::Error;

use super::lower;

/// A paged attention over `pool`: `o`, and the `lse` when it has one.
pub fn attend(ctx: &Ctx<'_>, call: Paged, pool: KvPool) -> Result<(), Error> {
    lower(ctx, |l| {
        let (call_v, pool_v) = call.try_map(&pool, |t| l.read(t))?;
        let tuning = l.tuning();
        let (o, lse) = paged::attend(l.f(), call_v, pool_v, tuning)?;
        l.land(call.o, o)?;
        if let (Some(t), Some(v)) = (call.lse, lse) {
            l.land(t, v)?;
        }
        Ok(())
    })
}

/// Folds a per-head sink into `o`, in place.
pub fn sink(
    ctx: &Ctx<'_>,
    o: Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    lower(ctx, |l| {
        let ov = l.read(o)?;
        let lv = l.read(lse)?;
        let sv = l.read(sink)?;
        let y = paged::sink(l.f(), ov, lv, sv, head_dim)?;
        l.land(o, y)
    })
}

/// Caps `x`'s logits, in place.
pub fn logit_softcap(ctx: &Ctx<'_>, x: Tensor, cap: f32) -> Result<(), Error> {
    lower(ctx, |l| {
        let v = l.read(x)?;
        let y = paged::logit_softcap(l.f(), v, cap)?;
        l.land(x, y)
    })
}

/// Merges two lse-carrying readings into `(o, lse)`.
pub fn merge_lse(
    ctx: &Ctx<'_>,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    lower(ctx, |l| {
        let a = l.read(o1)?;
        let b = l.read(o2)?;
        let l1 = l.read(lse1)?;
        let l2 = l.read(lse2)?;
        let (ov, lv) = kernels_xla::attn::merge::merge_lse(
            l.f(),
            a,
            l1,
            b,
            l2,
            heads,
            head_dim,
            o.landing(),
            lse.landing(),
        )?;
        l.land(o, ov)?;
        l.land(lse, lv)
    })
}

/// Files each row's `k` and `v` in `pool`'s keys and values planes.
pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    lower(ctx, |l| {
        let wp = l.read(write_page)?;
        let wo = l.read(write_offset)?;
        let kv = l.read(k)?;
        let keys = l.read(pool.keys)?;
        let vv = l.read(v)?;
        let values = l.read(pool.values)?;
        let (keys, values) = paged::kv_append(l.f(), kv, vv, keys, values, pool, wp, wo)?;
        l.land(pool.keys, keys)?;
        l.land(pool.values, values)
    })
}

/// Files each row of `plane` in both of `pool`'s planes.
pub fn kv_append_shared(
    ctx: &Ctx<'_>,
    plane: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    lower(ctx, |l| {
        let wp = l.read(write_page)?;
        let wo = l.read(write_offset)?;
        let p = l.read(plane)?;
        let keys = l.read(pool.keys)?;
        let values = l.read(pool.values)?;
        let (keys, values) = paged::kv_append_shared(l.f(), p, keys, values, pool, wp, wo)?;
        l.land(pool.keys, keys)?;
        l.land(pool.values, values)
    })
}

pub mod dense {
    use crate::emit::Ctx;
    use crate::handle::Tensor;
    use kernels_xla::Error;

    use super::super::lower;

    /// Bidirectional attention within each segment.
    pub fn bidirectional(
        ctx: &Ctx<'_>,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        segments: Tensor,
        head_dim: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let seg = l.read(segments)?;
            let qv = l.read(q)?;
            let kv = l.read(k)?;
            let vv = l.read(v)?;
            let tuning = l.tuning();
            let y = kernels_xla::attn::dense::bidirectional(
                l.f(),
                qv,
                kv,
                vv,
                seg,
                head_dim,
                sm_scale,
                o.landing(),
                tuning,
            )?;
            l.land(o, y)
        })
    }
}

pub mod ragged {
    use crate::emit::Ctx;
    use crate::handle::RaggedMask;
    use crate::handle::Tensor;
    use kernels_xla::Error;

    use super::super::lower;

    /// Ragged (segmented) attention under `mask`.
    pub fn forward(
        ctx: &Ctx<'_>,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_indptr: Tensor,
        kv_indptr: Tensor,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        mask: &RaggedMask,
        o: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let qi = l.read(q_indptr)?;
            let ki = l.read(kv_indptr)?;
            let mask = mask.try_map(|t| l.read(t))?;
            let qv = l.read(q)?;
            let kv = l.read(k)?;
            let vv = l.read(v)?;
            let tuning = l.tuning();
            let y = kernels_xla::attn::ragged::forward(
                l.f(),
                qv,
                kv,
                vv,
                qi,
                ki,
                head_dim,
                kv_heads,
                sm_scale,
                &mask,
                o.landing(),
                tuning,
            )?;
            l.land(o, y)
        })
    }
}

pub mod mla {
    use kernels_xla::attn::mla;

    use crate::emit::Ctx;
    use crate::handle::MlaCall;
    use crate::handle::{KvPool, Tensor};
    use kernels_xla::Error;

    use super::super::{Lower, lower};

    /// An absorbed MLA attention over the latent `pool`.
    pub fn flash(ctx: &Ctx<'_>, call: MlaCall, pool: KvPool) -> Result<(), Error> {
        lower(ctx, |l| {
            let (call_v, pool_v) = call.try_map(&pool, |t| l.read(t))?;
            let tuning = l.tuning();
            let o = mla::flash(l.f(), call_v, pool_v, tuning)?;
            l.land(call.o, o)
        })
    }

    fn land_latents(
        l: &mut Lower<'_, '_>,
        kv_c: Tensor,
        k_pe: Tensor,
        (c, pe): (kernels_xla::hlo::Val, Option<kernels_xla::hlo::Val>),
    ) -> Result<(), Error> {
        l.land(kv_c, c)?;
        match pe {
            Some(pe) => l.land(k_pe, pe),
            None => Ok(()),
        }
    }

    /// Splits and normalizes the compressed kv into `kv_c` and `k_pe`.
    pub fn latents(
        ctx: &Ctx<'_>,
        kv_a: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let a = l.read(kv_a)?;
            let w = l.read(weight)?;
            let out = mla::latents(
                l.f(),
                a,
                w,
                eps,
                kv_lora_rank,
                kv_c.landing(),
                k_pe.landing(),
            )?;
            land_latents(l, kv_c, k_pe, out)
        })
    }

    /// As [`latents`], `k_pe` roped at the row's position.
    pub fn latents_rope(
        ctx: &Ctx<'_>,
        kv_a: Tensor,
        positions: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let a = l.read(kv_a)?;
            let w = l.read(weight)?;
            let pos = l.read(positions)?;
            let out = mla::latents_rope(
                l.f(),
                a,
                w,
                pos,
                eps,
                kv_lora_rank,
                rope_dim,
                theta,
                kv_c.landing(),
                k_pe.landing(),
            )?;
            land_latents(l, kv_c, k_pe, out)
        })
    }

    /// Splits `q_b` into its `q_nope` and `q_pe` heads.
    pub fn split_q_b(
        ctx: &Ctx<'_>,
        q_b: Tensor,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        q_nope: Tensor,
        q_pe: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let v = l.read(q_b)?;
            let (a, b) = mla::split_q_b(
                l.f(),
                v,
                heads,
                nope_dim,
                rope_dim,
                q_nope.landing(),
                q_pe.landing(),
            )?;
            if let Some(a) = a {
                l.land(q_nope, a)?;
            }
            if let Some(b) = b {
                l.land(q_pe, b)?;
            }
            Ok(())
        })
    }

    /// Absorbs `kv_b`'s key part into the query.
    pub fn absorb_q(
        ctx: &Ctx<'_>,
        q_nope: Tensor,
        kv_b: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let q = l.read(q_nope)?;
            let w = l.read(kv_b)?;
            let y = mla::absorb_q(
                l.f(),
                q,
                w,
                heads,
                kv_lora_rank,
                nope_dim,
                v_head_dim,
                q_latent.landing(),
            )?;
            l.land(q_latent, y)
        })
    }

    /// Projects the latent reading out through `kv_b`'s value part.
    pub fn absorb_out(
        ctx: &Ctx<'_>,
        latent: Tensor,
        kv_b: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        o: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let lat = l.read(latent)?;
            let w = l.read(kv_b)?;
            let y = mla::absorb_out(
                l.f(),
                lat,
                w,
                heads,
                kv_lora_rank,
                v_head_dim,
                nope_dim,
                o.landing(),
            )?;
            l.land(o, y)
        })
    }

    /// Files each row's latent (and rope part, when `k_pe` has width) in
    /// the latent pool.
    pub fn kv_append(
        ctx: &Ctx<'_>,
        kv_c: Tensor,
        k_pe: Tensor,
        pool: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let wp = l.read(write_page)?;
            let wo = l.read(write_offset)?;
            let c = l.read(kv_c)?;
            let keys = l.read(pool.keys)?;
            let rope = if k_pe.width == 0 {
                None
            } else {
                Some((l.read(k_pe)?, l.read(pool.values)?))
            };
            let (keys, pe) = mla::kv_append(l.f(), c, keys, rope, pool, wp, wo)?;
            l.land(pool.keys, keys)?;
            match pe {
                Some(pe) => l.land(pool.values, pe),
                None => Ok(()),
            }
        })
    }
}

pub mod index {
    use crate::emit::Ctx;
    use crate::handle::{KvPool, Tensor};
    use kernels_xla::Error;
    use kernels_xla::attn::index;
    use kernels_xla::attn::pool::Entries;

    use super::super::{Lower, lower};

    /// The index pool's page tables, then (after `between`) its keys plane.
    fn planes<T>(
        l: &mut Lower<'_, '_>,
        pool: &KvPool,
        between: impl FnOnce(&mut Lower<'_, '_>) -> Result<T, Error>,
    ) -> Result<(Entries, T), Error> {
        let page_indices = l.read(pool.page_indices)?;
        let page_indptr = l.read(pool.page_indptr)?;
        let mid = between(l)?;
        let keys = l.read(pool.keys)?;
        Ok((
            Entries {
                page_indices,
                page_indptr,
                keys,
            },
            mid,
        ))
    }

    /// The index key layernormed and roped, in place.
    pub fn layernorm_rope(
        ctx: &Ctx<'_>,
        k: Tensor,
        positions: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let x = l.read(k)?;
            let w = l.read(weight)?;
            let b = l.read(bias)?;
            let pos = l.read(positions)?;
            let y = index::layernorm_rope(l.f(), x, pos, w, b, eps, rope_dim, theta)?;
            l.land(k, y)
        })
    }

    /// The index query roped, in place.
    pub fn rope(
        ctx: &Ctx<'_>,
        q: Tensor,
        positions: Tensor,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let x = l.read(q)?;
            let pos = l.read(positions)?;
            let y = index::rope(l.f(), x, pos, heads, head_dim, rope_dim, theta)?;
            l.land(q, y)
        })
    }

    /// Each row's `top_k` best-scoring index blocks, into `selection`; with
    /// `tokens`, the blocks read back as their tokens and the row's open
    /// block's (`pooled_tokens`).
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
        tokens: bool,
        selection: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let (planes, (qv, w, pos, req)) = planes(l, keys, |l| {
                let qv = l.read(q)?;
                let w = l.read_opt(weights)?;
                let pos = l.read(positions)?;
                let req = l.read(request_of_token)?;
                Ok((qv, w, pos, req))
            })?;
            let topk = if tokens {
                index::topk_tokens
            } else {
                index::topk
            };
            let y = topk(
                l.f(),
                qv,
                w,
                planes,
                keys,
                pos,
                req,
                heads,
                head_dim,
                top_k,
                ratio,
                selection.landing(),
            )?;
            l.land(selection, y)
        })
    }

    /// Each row's `top_k` blocks per head, ranked by their best key, into
    /// `selection` (`block_max`).
    pub fn topk_block_max(
        ctx: &Ctx<'_>,
        q: Tensor,
        keys: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        block: u32,
        selection: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let (planes, (qv, pos, req)) = planes(l, keys, |l| {
                Ok((l.read(q)?, l.read(positions)?, l.read(request_of_token)?))
            })?;
            let y = index::topk_block_max(
                l.f(),
                qv,
                planes,
                keys,
                pos,
                req,
                heads,
                head_dim,
                top_k,
                block,
                selection.landing(),
            )?;
            l.land(selection, y)
        })
    }

    /// Files each row's index key in the index pool.
    pub fn kv_append(
        ctx: &Ctx<'_>,
        k: Tensor,
        keys: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let wp = l.read(write_page)?;
            let wo = l.read(write_offset)?;
            let kv = l.read(k)?;
            let plane = l.read(keys.keys)?;
            let y = index::kv_append(l.f(), kv, plane, keys, wp, wo)?;
            l.land(keys.keys, y)
        })
    }

    /// Each closing row's block mean of index keys, into `entries`.
    pub fn block_mean(
        ctx: &Ctx<'_>,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        keys: &KvPool,
        head_dim: u32,
        ratio: u32,
        entries: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let (planes, (bpos, breq)) = planes(l, keys, |l| {
                Ok((l.read(boundary_pos)?, l.read(boundary_req)?))
            })?;
            let y = index::block_mean(
                l.f(),
                bpos,
                breq,
                planes,
                keys,
                head_dim,
                ratio,
                entries.landing(),
            )?;
            l.land(entries, y)
        })
    }
}

pub mod pool {
    use kernels_xla::attn::pool;

    use crate::emit::Ctx;
    use crate::handle::PoolLse;
    use crate::handle::{KvPool, RaggedTensor, Tensor};
    use kernels_xla::Error;

    use super::super::lower;

    /// A compressed-pool reading over `entries`: `o` and `lse`.
    pub fn lse(ctx: &Ctx<'_>, call: PoolLse, entries: KvPool) -> Result<(), Error> {
        lower(ctx, |l| {
            let (o, lse) = match call.try_map(&entries, |t| l.read(t))? {
                Some((call_v, planes)) => pool::lse(l.f(), call_v, planes, &entries)?,
                None => pool::nothing(l.f(), call.o.landing(), call.lse.landing()),
            };
            l.land(call.o, o)?;
            l.land(call.lse, lse)
        })
    }

    fn boundary(
        ctx: &Ctx<'_>,
        prefill: bool,
        positions: Tensor,
        request_of_token: Tensor,
        row_valid: Tensor,
        ratio: u32,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        boundary_rope: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let pos = l.read(positions)?;
            let req = l.read(request_of_token)?;
            let valid = l.read(row_valid)?;
            let kernel = if prefill {
                pool::boundary_prefill
            } else {
                pool::boundary_decode
            };
            let (p, r, o) = kernel(
                l.f(),
                pos,
                req,
                valid,
                ratio,
                boundary_pos.landing(),
                boundary_req.landing(),
                boundary_rope.landing(),
            )?;
            l.land(boundary_pos, p)?;
            l.land(boundary_req, r)?;
            l.land(boundary_rope, o)
        })
    }

    /// Marks the decode rows closing a `ratio` block.
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
            false,
            positions,
            request_of_token,
            row_valid,
            ratio,
            boundary_pos,
            boundary_req,
            boundary_rope,
        )
    }

    /// Marks the prompt rows closing a `ratio` block.
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
            true,
            positions.data,
            request_of_token,
            row_valid,
            ratio,
            boundary_pos,
            boundary_req,
            boundary_rope,
        )
    }

    /// Files each row's compressor projections in the state planes.
    pub fn state_write(
        ctx: &Ctx<'_>,
        kv: Tensor,
        score: Tensor,
        pages: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
        head_dim: u32,
        ratio: u32,
        state_kv: Tensor,
        state_score: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let wp = l.read(write_page)?;
            let wo = l.read(write_offset)?;
            let k = l.read(kv)?;
            let s = l.read(score)?;
            let sk = l.read(state_kv)?;
            let ss = l.read(state_score)?;
            let (kv, sc) = pool::state_write(l.f(), k, s, pages, wp, wo, head_dim, ratio, sk, ss)?;
            l.land(state_kv, kv)?;
            l.land(state_score, sc)
        })
    }

    /// Each closing row's pooled block entry, into `entries`.
    pub fn gather(
        ctx: &Ctx<'_>,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        pages: &KvPool,
        head_dim: u32,
        ratio: u32,
        state_kv: Tensor,
        state_score: Tensor,
        ape: Option<Tensor>,
        entries: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let page_indices = l.read(pages.page_indices)?;
            let page_indptr = l.read(pages.page_indptr)?;
            let bpos = l.read(boundary_pos)?;
            let breq = l.read(boundary_req)?;
            let sk = l.read(state_kv)?;
            let ss = l.read(state_score)?;
            let ape = l.read_opt(ape)?;
            let y = pool::gather(
                l.f(),
                bpos,
                breq,
                page_indices,
                page_indptr,
                pages,
                head_dim,
                ratio,
                sk,
                ss,
                ape,
                entries.landing(),
            )?;
            l.land(entries, y)
        })
    }

    /// Files each closing row's entry in the compressed pool's keys plane.
    pub fn kv_append(
        ctx: &Ctx<'_>,
        entries: Tensor,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        into: &KvPool,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let page_indices = l.read(into.page_indices)?;
            let page_indptr = l.read(into.page_indptr)?;
            let bpos = l.read(boundary_pos)?;
            let breq = l.read(boundary_req)?;
            let e = l.read(entries)?;
            let keys = l.read(into.keys)?;
            let y = pool::kv_append(l.f(), e, bpos, breq, page_indices, page_indptr, keys, into)?;
            l.land(into.keys, y)
        })
    }
}
