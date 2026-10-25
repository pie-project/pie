use kernels_xla::gather;
use poem_exec::{DispatchGather, KernelError};
use poem_ir::Gather;

use poem_ir::Operands;

use crate::run::Run;

impl DispatchGather for Run<'_> {
    fn dispatch(&mut self, op: &Gather) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.gather(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn gather(&mut self, op: &Gather) -> Result<(), kernels_xla::Error> {
        match op {
            Gather::NgramIds {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } => {
                let pool = self.recurrent(*state);
                let out = self.tensor(*ngram_ids);
                crate::lower::lower(self.ctx(), |l| {
                    let ids = l.read(self.tensor(*ids))?;
                    let slab = l.read(pool.state)?;
                    let map = l.read_opt(map.map(|m| self.tensor(m)))?;
                    let slots = l.read(pool.slots)?;
                    let (rows, next) = gather::ngram::ngram_ids(
                        l.f(),
                        ids,
                        slab,
                        slots,
                        *eos,
                        mults,
                        primes,
                        offsets,
                        *heads_per_ngram,
                        map,
                        out.landing(),
                    )?;
                    l.land(out, rows)?;
                    l.land(pool.state, next)
                })
            }
            Gather::NgramIdsChunked {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } => {
                let pool = self.recurrent(*state);
                let ids = self.ragged(*ids);
                let out = self.tensor(*ngram_ids);
                crate::lower::lower(self.ctx(), |l| {
                    let data = l.read(ids.data)?;
                    let slab = l.read(pool.state)?;
                    let map = l.read_opt(map.map(|m| self.tensor(m)))?;
                    let indptr = l.read(ids.indptr)?;
                    let slots = l.read(pool.slots)?;
                    let (rows, next) = gather::ngram::ngram_ids_chunked(
                        l.f(),
                        kernels_xla::tensor::RaggedTensor { data, indptr },
                        slab,
                        slots,
                        *eos,
                        mults,
                        primes,
                        offsets,
                        *heads_per_ngram,
                        map,
                        out.landing(),
                    )?;
                    l.land(out, rows)?;
                    l.land(pool.state, next)
                })
            }
            Gather::Embed {
                ids,
                table,
                vocab,
                y,
                weights: None,
                scales: None,
                concat: false,
                ..
            } => match self.banked(*table) {
                Some(bank) => crate::lower::lower(self.ctx(), |l| {
                    let ids_v = l.read(self.tensor(*ids))?;
                    let table_v = l.bank(bank)?;
                    let o_y = self.tensor(*y);
                    let v =
                        gather::embed_gather_mb_4bit(l.f(), ids_v, table_v, *vocab, o_y.landing())?;
                    l.land(o_y, v)
                }),
                None if self.tensor(*table).rows < *vocab => crate::lower::lower(self.ctx(), |l| {
                    let ids_v = l.read(self.tensor(*ids))?;
                    let table_v = l.read(self.tensor(*table))?;
                    let o_y = self.tensor(*y);
                    let v = gather::embed_vocab_shard(
                        l.f(),
                        ids_v,
                        table_v,
                        self.rank(),
                        o_y.landing(),
                    )?;
                    l.land(o_y, v)
                }),
                None => crate::lower::lower(self.ctx(), |l| {
                    let ids_v = l.read(self.tensor(*ids))?;
                    let table_v = l.read(self.tensor(*table))?;
                    let o_y = self.tensor(*y);
                    let v = gather::embed(l.f(), ids_v, table_v, *vocab, o_y.landing())?;
                    l.land(o_y, v)
                }),
            },

            Gather::Embed {
                scales: Some(_),
                weights: None,
                concat: true,
                ..
            } => Err(kernels_xla::Error::Unsupported {
                op: "gather.embed(mxfp8)",
            }),
            Gather::Embed {
                ids,
                table,
                scales: None,
                vocab,
                y,
                weights: None,
                concat: true,
            } => match self.banked(*table) {
                Some(bank) => {
                    let ids = self.tensor(*ids);
                    let width = self.tensor(*y).width / ids.width.max(1);
                    let guard = gather::table_rows(&bank, width).min(*vocab);
                    crate::lower::lower(self.ctx(), |l| {
                        let ids_v = l.read(ids)?;
                        let table_v = l.bank(bank)?;
                        let o_y = self.tensor(*y);
                        let v = gather::embed_concat_mb_4bit(
                            l.f(),
                            ids_v,
                            table_v,
                            guard,
                            o_y.landing(),
                        )?;
                        l.land(o_y, v)
                    })
                }
                None => {
                    let plane = self.tensor(*table);
                    let guard = plane.rows.min(*vocab);
                    crate::lower::lower(self.ctx(), |l| {
                        let ids_v = l.read(self.tensor(*ids))?;
                        let table_v = l.read(plane)?;
                        let o_y = self.tensor(*y);
                        let v = gather::embed_concat(l.f(), ids_v, table_v, guard, o_y.landing())?;
                        l.land(o_y, v)
                    })
                }
            },
            Gather::SplitQkv {
                packed,
                q_width,
                kv_width,
                q,
                k,
                v,
            } => crate::lower::lower(self.ctx(), |l| {
                let packed_v = l.read(self.tensor(*packed))?;
                let o_q = self.tensor(*q);
                let o_k = self.tensor(*k);
                let o_v = self.tensor(*v);
                let (v0, v1, v2) = gather::split_qkv(
                    l.f(),
                    packed_v,
                    *q_width,
                    *kv_width,
                    o_q.landing(),
                    o_k.landing(),
                    o_v.landing(),
                )?;
                l.land(o_q, v0)?;
                l.land(o_k, v1)?;
                l.land(o_v, v2)?;
                Ok(())
            }),
            Gather::SplitQGate {
                packed,
                head_dim,
                q,
                gate,
            } => crate::lower::lower(self.ctx(), |l| {
                let packed_v = l.read(self.tensor(*packed))?;
                let o_q = self.tensor(*q);
                let o_gate = self.tensor(*gate);
                let (v0, v1) = gather::split_q_gate(
                    l.f(),
                    packed_v,
                    *head_dim,
                    o_q.landing(),
                    o_gate.landing(),
                )?;
                l.land(o_q, v0)?;
                l.land(o_gate, v1)?;
                Ok(())
            }),
            Gather::SplitRows {
                x,
                width,
                left,
                right,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_left = self.tensor(*left);
                let o_right = self.tensor(*right);
                let (v0, v1) =
                    gather::split_rows(l.f(), x_v, *width, o_left.landing(), o_right.landing())?;
                l.land(o_left, v0)?;
                l.land(o_right, v1)?;
                Ok(())
            }),

            Gather::PoolRows { x, side, y } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = gather::pool_rows(l.f(), x_v, *side, o_y.landing())?;
                l.land_head(o_y, v)
            }),

            Gather::MergeRows { x, side, y } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = gather::merge_rows(l.f(), x_v, *side, o_y.landing())?;
                l.land_head(o_y, v)
            }),

            Gather::Pixels {
                x,
                patch,
                mean,
                std,
                order,
                temporal,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                if let Some(v) = gather::pixels(
                    l.f(),
                    x_v,
                    *patch,
                    *mean,
                    *std,
                    *order == poem_ir::PixelOrder::Chw,
                    *temporal,
                    o_y.landing(),
                )? {
                    l.land_head(o_y, v)?;
                }
                Ok(())
            }),

            Gather::GroupRows {
                csr,
                groups,
                lanes,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let csr_v = l.read(self.tensor(*csr))?;
                let groups_v = l.read(self.tensor(*groups))?;
                let lanes_v = l.read(self.tensor(*lanes))?;
                let o_y = self.tensor(*y);
                let v = kernels_xla::conv::rule::group_rows(
                    l.f(),
                    csr_v,
                    groups_v,
                    lanes_v,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Gather::ConcatCols { .. } | Gather::SplitQkvPadded { .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
            Gather::GridTaps {
                positions,
                grids,
                segments,
                kind,
                side,
                ids,
                weights,
            } => crate::lower::lower(self.ctx(), |l| {
                let positions_v = l.read(self.tensor(*positions))?;
                let (grids, segments) = (self.tensor(*grids), self.tensor(*segments));
                let bilinear = if *kind == poem_ir::TapKind::Bilinear {
                    let segments = l.read(segments)?;
                    Some(gather::Bilinear {
                        segments,
                        grids: l.read(grids)?,
                    })
                } else {
                    None
                };
                let o_ids = self.tensor(*ids);
                let o_weights = self.tensor(*weights);
                if let Some((v0, v1)) = gather::grid_taps(
                    l.f(),
                    positions_v,
                    grids,
                    segments,
                    bilinear,
                    *side,
                    o_ids.landing(),
                    o_weights.landing(),
                )? {
                    l.land_head(o_ids, v0)?;
                    l.land_head(o_weights, v1)?;
                }
                Ok(())
            }),

            Gather::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let routes_v = l.read(self.tensor(*routes))?;
                let src_v = l.read(self.tensor(*src))?;
                let y_v = l.read(self.uncut(*y))?;
                let o_y = self.uncut(*y);
                let v = gather::scatter_live_rows(l.f(), src_v, routes_v, y_v)?;
                l.land(o_y, v)
            }),

            Gather::Embed {
                ids,
                weights: Some(weights),
                table,
                vocab,
                y,
                scales: None,
                concat: false,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let table_v = l.read(self.tensor(*table))?;
                let ids_v = l.read(self.tensor(*ids))?;
                let weights_v = l.read(self.tensor(*weights))?;
                let o_y = self.tensor(*y);
                let v = gather::embed_weighted(
                    l.f(),
                    ids_v,
                    weights_v,
                    table_v,
                    *vocab,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),

            // The forms no builder makes.
            Gather::Embed { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Gather::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let index_v = l.read(self.tensor(*routes))?;
                let tight_v = l.read(self.tensor(*src))?;
                let wide_v = l.read(self.uncut(*y))?;
                let o_wide = self.uncut(*y);
                let v = gather::scatter_rows(l.f(), tight_v, index_v, wide_v)?;
                l.land(o_wide, v)
            }),

            Gather::PackRows { x, perm, y } => crate::lower::lower(self.ctx(), |l| {
                let perm_v = l.read(self.tensor(*perm))?;
                let x_v = l.read(self.uncut(*x))?;
                let o_y = self.tensor(*y);
                let v = gather::pack_rows(l.f(), x_v, perm_v, o_y.landing())?;
                l.land(o_y, v)
            }),
            Gather::UnpackRows { x, perm, y } => crate::lower::lower(self.ctx(), |l| {
                let perm_v = l.read(self.tensor(*perm))?;
                let x_v = l.read(self.tensor(*x))?;
                let y_v = l.read(self.uncut(*y))?;
                let o_y = self.uncut(*y);
                let v = gather::unpack_rows(l.f(), x_v, perm_v, y_v)?;
                l.land(o_y, v)
            }),
            Gather::Rows { x, rows, y } => crate::lower::lower(self.ctx(), |l| {
                let index_v = l.read(self.tensor(*rows))?;
                let wide_v = l.read(self.uncut(*x))?;
                let o_tight = self.tensor(*y);
                let v = gather::gather_rows(l.f(), wide_v, index_v, o_tight.landing())?;
                l.land(o_tight, v)
            }),
            Gather::Select {
                table,
                layer,
                width,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let table_v = l.read(self.tensor(*table))?;
                let o_y = self.tensor(*y);
                let v = gather::select(l.f(), table_v, *layer, *width, o_y.landing())?;
                l.land(o_y, v)
            }),
        }
    }
}
