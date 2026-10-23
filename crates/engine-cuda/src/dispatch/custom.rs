use kernels_cuda::attn::qkv_append;
use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};
use poem_ir::{NormAxis, NormWeight};

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        match op {
            Fused::QkvFusedQknormRopeVnormWrite { .. } => self.custom_cuda(op, None),
            Fused::SplitQkvFused { rest, qkv, .. } => {
                let rest = self.tensor(*rest);
                self.custom_cuda(qkv, Some(rest))
            }
            Fused::GdnStep { .. } | Fused::QGateNormRopeAppend { .. } => {
                Err(kernels_cuda::Error::Unsupported { op: op.name() })
            }
            Fused::RopeYarnAppend { .. } => self.rope_yarn_append(op),
            Fused::NormRopeLastAppend { .. } => self.norm_rope_last_append(op),
            Fused::PoolCompress { .. } => self.pool_compress(op),
            Fused::NormRopePair { .. } => self.norm_rope_pair(op),
            Fused::IndexTopkRope { .. } => self.index_topk_rope(op),
            Fused::SplitRopeYarnAppend { .. } => self.split_rope_yarn_append(op),
            Fused::QgQknormRopeAppend { .. } => self.qg_qknorm_rope_append(op),
            Fused::LseSink { .. } => self.lse_sink(op),
            Fused::PoolAttention { .. } => self.pool_attention(op),
            Fused::MlaPrefillExpanded { .. } => self.mla_prefill_expanded(op),
            Fused::HcPrenormProject { .. } => self.hc_prenorm_project(op),
            Fused::HcGatesNorm { .. } => self.hc_gates_norm(op),
            Fused::ResBlendRmsnorm { .. } => self.res_blend_rmsnorm(op),
            Fused::HcPre { .. } => self.hc_pre(op),
            Fused::MatmulGeglu { .. }
            | Fused::MatmulSwiglu { .. }
            | Fused::MlpSwiglu { .. }
            | Fused::LmHeadSoftcap { .. }
            | Fused::MatmulBias { .. }
            | Fused::MatmulSplitRows { .. }
            | Fused::MoeSelectGlu { .. }
            | Fused::MoeExperts { .. } => self.fused_linear(op),
            Fused::ResidualAddRmsnorm { .. }
            | Fused::RmsnormResidualAdd { .. }
            | Fused::EmbedScaleAdd { .. }
            | Fused::EmbedScaleAddSelect { .. }
            | Fused::RmsnormRopePartialQ { .. }
            | Fused::HcNormProject { .. }
            | Fused::NormRopeLast { .. }
            | Fused::NormModulate { .. }
            | Fused::GatedResidualNormModulate { .. } => {
                let mut outputs = Vec::new();
                op.outputs(&mut outputs);
                if outputs.first().is_some_and(|out| self.lane_shaped(*out)) {
                    self.unseated(|| self.fused_elemwise(op))
                } else {
                    self.fused_elemwise(op)
                }
            }
        }
        .map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The e4m3 1x128 planes a sublayer's pre-mix lands its normed row `y`
    /// in beside the bf16 one, for the GEMM that quantizes it next
    /// (`Run::hold_e4m3`): held only where the pre-mix will land them.
    fn hc_pre_held(
        &mut self,
        y: poem_ir::ValueId,
        emits: bool,
    ) -> Result<Option<[u64; 2]>, kernels_cuda::Error> {
        #[cfg(feature = "cuda")]
        if emits {
            return Ok(Some(self.hold_e4m3(y)?));
        }
        let _ = (y, emits);
        Ok(None)
    }

    /// A sublayer's whole pre-mix: two launches where the profile's tiled
    /// hyper-connection kernels run (unread rows left unwritten), else as traced.
    fn hc_pre(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::HcPre {
            norm,
            project,
            gates,
            rmsnorm,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        // A prefill's rows: the one-pass projection, then the gates and the
        // norm apart, whose 16-byte forms stream a long fire faster than the
        // block-a-row pre-mix.
        if self.ctx().tuning().hc_tile
            && let poem_ir::Elemwise::HcRmsnormF32 { streams, eps, .. } = &**norm
            && let poem_ir::Elemwise::HcProject {
                weight,
                stream_count,
                mixes,
                ..
            } = &**project
            && self.tensor(*streams).rows >= kernels_cuda::elemwise::hc::PRE_WIDE_FROM
            && kernels_cuda::elemwise::hc::prenorm_project(
                self.ctx(),
                self.tensor(*streams),
                *eps,
                self.tensor(*weight),
                *stream_count,
                &mut self.tensor(*mixes),
            )?
        {
            self.elemwise(gates)?;
            // The normed row lands its e4m3 1x128 operand beside it for the
            // GEMMs that read it next, where the profile says
            // (`kernel.elemwise.norm_emit`).
            #[cfg(feature = "cuda")]
            if self.ctx().tuning().norm_emit
                && let poem_ir::Elemwise::Rmsnorm {
                    x,
                    weight: NormWeight::Scale(weight),
                    eps,
                    y,
                    axis: NormAxis::Row,
                } = &**rmsnorm
                && self.tensor(*y).width == 4096
            {
                let planes = self.hold_e4m3(*y)?;
                if kernels_cuda::linear::act_quant::rmsnorm_act_quant_e4m3(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*weight),
                    *eps,
                    &mut self.tensor(*y),
                    planes,
                )? {
                    return Ok(());
                }
                self.drop_held();
            }
            return self.elemwise(rmsnorm);
        }
        if self.ctx().tuning().hc_tile
            && let (
                poem_ir::Elemwise::HcRmsnormF32 {
                    streams,
                    eps: mix_eps,
                    ..
                },
                poem_ir::Elemwise::HcProject {
                    weight: plane,
                    stream_count,
                    mixes,
                    ..
                },
                poem_ir::Elemwise::HcGates {
                    scale,
                    base,
                    gate_eps,
                    alpha,
                    sinkhorn,
                    comb: true,
                    post_mix,
                    comb_mix,
                    ..
                },
                poem_ir::Elemwise::Rmsnorm {
                    weight: NormWeight::Scale(weight),
                    eps,
                    y,
                    axis: NormAxis::Row,
                    ..
                },
            ) = (&**norm, &**project, &**gates, &**rmsnorm)
            && let emits = kernels_cuda::elemwise::hc::pre_emits(
                &self.tensor(*streams),
                &self.tensor(*plane),
                &self.tensor(*mixes),
                *stream_count,
                &self.tensor(*weight),
                &self.tensor(*y),
            )
            && let held = self.hc_pre_held(*y, emits)?
            && kernels_cuda::elemwise::hc::pre(
                self.ctx(),
                self.tensor(*streams),
                *mix_eps,
                self.tensor(*plane),
                *stream_count,
                &mut self.tensor(*mixes),
                kernels_cuda::elemwise::hc::Gate {
                    scale: self.tensor(*scale),
                    base: self.tensor(*base),
                    gate_eps: *gate_eps,
                    alpha: *alpha,
                    sinkhorn: *sinkhorn,
                },
                self.tensor(*weight),
                *eps,
                held,
                &mut self.tensor(*post_mix),
                &mut self.tensor(*comb_mix),
                &mut self.tensor(*y),
            )?
        {
            return Ok(());
        }
        self.elemwise(norm)?;
        self.elemwise(project)?;
        self.elemwise(gates)?;
        self.elemwise(rmsnorm)
    }

    /// AttnRes's blend and the norm of its row: in one pass where the
    /// profile says (`kernel.attn_res.fused_norm`) and the row takes it,
    /// else the two.
    fn res_blend_rmsnorm(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::ResBlendRmsnorm { blend, norm } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        if let (
            poem_ir::Elemwise::ResBlend {
                prefix,
                blocks,
                weight: blend_weight,
                eps: blend_eps,
                proj,
                ..
            },
            poem_ir::Elemwise::Rmsnorm {
                weight: NormWeight::Scale(weight),
                eps,
                y,
                axis: NormAxis::Row,
                ..
            },
        ) = (&**blend, &**norm)
            && self.ctx().tuning().attn_res_fused_norm
        {
            let blocks: Vec<kernels_cuda::tensor::Tensor> =
                blocks.iter().map(|b| self.tensor(*b)).collect();
            if kernels_cuda::elemwise::modulate::res_blend_norm(
                self.ctx(),
                self.tensor(*prefix),
                &blocks,
                self.tensor(*blend_weight),
                *blend_eps,
                self.tensor(*proj),
                self.tensor(*weight),
                *eps,
                &mut self.tensor(*y),
            )? {
                return Ok(());
            }
        }
        self.elemwise(blend)?;
        self.elemwise(norm)
    }

    fn hc_gates_norm(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::HcGatesNorm { gates, norm } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let (
            poem_ir::Elemwise::HcGates {
                normed,
                streams,
                scale,
                base,
                stream_count,
                gate_eps,
                alpha,
                sinkhorn,
                comb: true,
                post_mix,
                comb_mix,
                ..
            },
            poem_ir::Elemwise::Rmsnorm {
                weight: NormWeight::Scale(weight),
                eps,
                y,
                axis: NormAxis::Row,
                ..
            },
        ) = (&**gates, &**norm)
        else {
            self.elemwise(gates)?;
            return self.elemwise(norm);
        };
        if self.ctx().tuning().hc_tile {
            let landed = kernels_cuda::elemwise::hc::gates_norm(
                self.ctx(),
                self.tensor(*normed),
                self.tensor(*streams),
                *stream_count,
                kernels_cuda::elemwise::hc::Gate {
                    scale: self.tensor(*scale),
                    base: self.tensor(*base),
                    gate_eps: *gate_eps,
                    alpha: *alpha,
                    sinkhorn: *sinkhorn,
                },
                self.tensor(*weight),
                *eps,
                &mut self.tensor(*post_mix),
                &mut self.tensor(*comb_mix),
                &mut self.tensor(*y),
            )?;
            if landed {
                return Ok(());
            }
        }
        self.elemwise(gates)?;
        self.elemwise(norm)
    }

    /// A stream norm and the mix projection over it: one pass where the profile's
    /// tiled hyper-connection kernels run (normed rows unwritten), else the two.
    fn hc_prenorm_project(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::HcPrenormProject { norm, project } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let (
            poem_ir::Elemwise::HcRmsnormF32 { streams, eps, .. },
            poem_ir::Elemwise::HcProject {
                weight,
                stream_count,
                mixes,
                ..
            },
        ) = (&**norm, &**project)
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        if self.ctx().tuning().hc_tile {
            let landed = kernels_cuda::elemwise::hc::prenorm_project(
                self.ctx(),
                self.tensor(*streams),
                *eps,
                self.tensor(*weight),
                *stream_count,
                &mut self.tensor(*mixes),
            )?;
            if landed {
                return Ok(());
            }
        }
        self.elemwise(norm)?;
        self.elemwise(project)
    }

    /// A latent attention's prefill, expanded (a KV head a query head, as vLLM's
    /// MLA prefill) from `kernel.mla.expanded_from` rows where it fits, else as traced.
    fn mla_prefill_expanded(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::MlaPrefillExpanded {
            split,
            absorb_q,
            prefill,
            absorb_out,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        #[cfg(feature = "cuda")]
        if let (
            poem_ir::Attn::MlaAbsorbQ {
                q_nope,
                kv_b,
                heads,
                kv_lora_rank,
                nope_dim,
                v_head_dim,
                ..
            },
            poem_ir::Attn::MlaPrefill {
                q,
                q_pe,
                cache,
                window: None,
                sm_scale,
                lse: None,
                ..
            },
            poem_ir::Attn::MlaAbsorbOut { o, .. },
        ) = (&**absorb_q, &**prefill, &**absorb_out)
        {
            use kernels_cuda::attn::mla::{self, ExpandedQ, Heads};
            let mut out = self.tensor(*o);
            let q_pe_t = self.tensor(*q_pe);
            let h = Heads {
                heads: *heads,
                kv_lora_rank: *kv_lora_rank,
                nope: *nope_dim,
                rope: q_pe_t.width / (*heads).max(1),
                v: *v_head_dim,
            };
            if out.rows >= self.ctx().tuning().mla_expanded_from && mla::expanded_takes(&h) {
                let queries = match split.as_deref() {
                    Some(poem_ir::Attn::MlaSplitQB { q_b, .. }) => {
                        ExpandedQ::Packed(self.tensor(*q_b))
                    }
                    _ => ExpandedQ::Split {
                        q_nope: self.tensor(*q_nope),
                        q_pe: q_pe_t,
                    },
                };
                let kv_b = self.dense_or_decoded(
                    "fused.mla_prefill_expanded",
                    *kv_b,
                    heads * (nope_dim + v_head_dim),
                    *kv_lora_rank,
                )?;
                let qo_indptr = self.ragged(*q).indptr;
                let pool = self.pool(*cache);
                if mla::prefill_expanded(
                    self.ctx(),
                    queries,
                    qo_indptr,
                    &pool,
                    kv_b,
                    h,
                    *sm_scale,
                    &mut out,
                )? {
                    return Ok(());
                }
            }
        }
        if let Some(split) = split {
            self.attention(split)?;
        }
        self.attention(absorb_q)?;
        self.attention(prefill)?;
        self.attention(absorb_out)
    }

    /// DeepSeek-V4's attention, then the turn of its output's tail where the
    /// fused op holds one and the pass did not land it.
    fn pool_attention(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let turned = self.pool_attention_core(op)?;
        if let Fused::PoolAttention {
            rotate: Some(rotate),
            ..
        } = op
            && !turned
        {
            self.elemwise(rotate)?;
        }
        Ok(())
    }

    /// DeepSeek-V4's attention: the window, the pool, their merge and the
    /// sinks, as the trace had them. The sinked output is the merge's plane.
    /// Whether the pass also turned the output's tail (`rotate`).
    fn pool_attention_core(&mut self, op: &Fused) -> Result<bool, kernels_cuda::Error> {
        let Fused::PoolAttention {
            window,
            pool,
            merge,
            sink: sinked,
            rotate,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let poem_ir::Attn::Sink {
            sink,
            head_dim,
            o_out,
            ..
        } = &**sinked
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        // The output's tail turned back with the output where the turn is
        // DeepSeek-V4's (interleaved, inverse, at the pool's positions).
        let turn = match (rotate.as_deref(), &**pool) {
            (
                Some(poem_ir::Elemwise::RopePartialLast {
                    q,
                    positions: at,
                    rotary_dim,
                    head_dim: d,
                    theta,
                    interleaved: true,
                    inverse: true,
                    yarn,
                    ..
                }),
                poem_ir::Attn::PoolLse { positions, .. }
                | poem_ir::Attn::PoolLseSelected { positions, .. },
            ) if q == o_out && at == positions && d == head_dim => {
                Some(kernels_cuda::attn::compressed::decode::Turn {
                    dim: *rotary_dim,
                    theta: *theta,
                    yarn: yarn.map(|y| kernels_cuda::elemwise::rope::Yarn {
                        factor: y.factor,
                        beta_fast: y.beta_fast,
                        beta_slow: y.beta_slow,
                        original_max_position: y.original_max_position,
                    }),
                })
            }
            _ => None,
        };
        // A window held above a replay floor the profile may raise
        // (`kernel.sdpa.compressed_replay`): only the fused kernels read the floor.
        let floored = self.ctx().tuning().compressed_replay
            && matches!(**window, poem_ir::Attn::PrefillLse { floor: Some(_), .. });
        // A decode's rows, up to what the profile says
        // (`kernel.sdpa.compressed_decode_until`): the four in one launch.
        let rows = self.tensor(*o_out).rows;
        // V4.1's FP4 cache from the rows the profile says
        // (`kernel.sdpa.compressed_fp4_pass_from`): the tcgen05 pass.
        let fp4_pass = self.fp4_pass(window, rows);
        if rows <= self.ctx().tuning().compressed_decode_until
            && !fp4_pass
            && let Some(landed) = self.dsv4_decode(window, Some(pool), *sink, *o_out, turn)
        {
            return landed.map(|()| turn.is_some());
        }
        // A prefill's rows, where the profile says
        // (`kernel.sdpa.compressed_prefill_from`): the four in one tcgen05 pass.
        #[cfg(feature = "cuda")]
        if let (
            poem_ir::Attn::PrefillLse {
                q,
                cache,
                window: span,
                head_dim: d,
                sm_scale,
                floor,
                ..
            },
            poem_ir::Attn::PoolLse {
                positions,
                request_of_token,
                entries,
                ratio,
                heads,
                ..
            }
            | poem_ir::Attn::PoolLseSelected {
                positions,
                request_of_token,
                entries,
                ratio,
                heads,
                ..
            },
        ) = (&**window, &**pool)
        {
            use kernels_cuda::attn::compressed::prefill::{self, Pooled};
            let from = self.ctx().tuning().compressed_prefill_from;
            let mut out = self.tensor(*o_out);
            // Its request ids index the whole fire's page tables too.
            if (fp4_pass || from > 0 && out.rows >= from)
                && let (Some(window_pool), Some(entries)) =
                    (self.pool_whole(*cache), self.pool_whole(*entries))
            {
                let selection = match &**pool {
                    poem_ir::Attn::PoolLseSelected {
                        selection, top_k, ..
                    } => Some((self.tensor(*selection), *top_k)),
                    _ => None,
                };
                let pooled = Pooled {
                    entries: &entries,
                    ratio: *ratio,
                    selection,
                };
                if prefill::takes(*heads, *d, *span, &window_pool, &pooled) {
                    return prefill::prefill(
                        self.ctx(),
                        op.name(),
                        self.tensor(*q),
                        self.tensor(*positions),
                        self.tensor(*request_of_token),
                        floor.map(|floor| self.tensor(floor)),
                        self.recorded(),
                        &window_pool,
                        &pooled,
                        self.tensor(*sink),
                        *sm_scale,
                        turn,
                        &mut out,
                    )
                    .map(|()| turn.is_some());
                }
            }
        }
        // DeepSeek's fp8 rows have no other reader, nor has a floored
        // window: the one launch at any width, or a refusal, never the four
        // ops reading them as bf16 (or past the floor).
        if self.fp8_rows(window) || floored {
            return self
                .dsv4_decode(window, Some(pool), *sink, *o_out, turn)
                .unwrap_or_else(|| Err(fp8_unread(op.name())))
                .map(|()| turn.is_some());
        }
        self.attention(window)?;
        self.attention(pool)?;
        self.attention(merge)?;
        self.attention(sinked).map(|()| false)
    }

    /// Whether `window` reads V4.1's FP4 cache at `rows` the profile sends
    /// to the tcgen05 pass (`kernel.sdpa.compressed_fp4_pass_from`).
    fn fp4_pass(&self, window: &poem_ir::Attn, rows: u32) -> bool {
        let from = self.ctx().tuning().compressed_fp4_pass_from;
        from > 0
            && rows >= from
            && matches!(*window, poem_ir::Attn::PrefillLse { cache, .. }
                if matches!(kernels_cuda::attn::compressed::decode::rows_of(&self.pool(cache)),
                    Some(kernels_cuda::attn::compressed::decode::Rows::Mx8)))
    }

    /// Whether `window`'s cache holds DeepSeek's quantized rows.
    fn fp8_rows(&self, window: &poem_ir::Attn) -> bool {
        match *window {
            poem_ir::Attn::PrefillLse { cache, .. } => {
                kernels_cuda::attn::compressed::decode::ds_rows(&self.pool(cache))
            }
            _ => false,
        }
    }

    /// The fire's runtime input `want` asks for, if the trace reads it.
    fn input_of(&self, want: &poem_ir::RuntimeInput) -> Option<poem_ir::ValueId> {
        self.values()
            .iter()
            .position(|decl| matches!(&decl.def, poem_ir::Def::Input(held) if held == want))
            .map(|at| poem_ir::ValueId(at as u32))
    }

    /// DeepSeek-V4's window (`PrefillLse`), pool if any (`PoolLse[Selected]`) and
    /// sinks in one launch (`attn::compressed::decode::decode`) into `o_out`; `None` where the
    /// shape doesn't fit or the region reads gathered seats.
    fn dsv4_decode(
        &mut self,
        window: &poem_ir::Attn,
        pool: Option<&poem_ir::Attn>,
        sink: poem_ir::ValueId,
        o_out: poem_ir::ValueId,
        turn: Option<kernels_cuda::attn::compressed::decode::Turn>,
    ) -> Option<Result<(), kernels_cuda::Error>> {
        use kernels_cuda::attn::compressed::decode;
        let poem_ir::Attn::PrefillLse {
            q,
            cache,
            window: Some(span),
            head_dim,
            sm_scale,
            floor,
            ..
        } = *window
        else {
            return None;
        };
        let (positions, request_of_token, pooled) = match pool {
            Some(
                poem_ir::Attn::PoolLse {
                    positions,
                    request_of_token,
                    entries,
                    ratio,
                    ..
                }
                | poem_ir::Attn::PoolLseSelected {
                    positions,
                    request_of_token,
                    entries,
                    ratio,
                    ..
                },
            ) => {
                let selection = match pool {
                    Some(poem_ir::Attn::PoolLseSelected {
                        selection, top_k, ..
                    }) => Some((self.tensor(*selection), *top_k)),
                    _ => None,
                };
                (
                    *positions,
                    *request_of_token,
                    Some((self.pool_whole(*entries)?, *ratio, selection)),
                )
            }
            Some(_) => return None,
            None => (
                self.input_of(&poem_ir::RuntimeInput::Positions)?,
                self.input_of(&poem_ir::RuntimeInput::Geometry {
                    space: 0,
                    kind: poem_ir::GeomKind::RequestOfToken,
                })?,
                None,
            ),
        };
        let q = self.tensor(q);
        let heads = q.width / head_dim.max(1);
        // The kernel reads each row's request id against the whole fire's
        // page tables, so it reads them whole whatever lanes this region
        // starts at (and not at all where the region reads gathered seats).
        let window_pool = self.pool_whole(cache)?;
        let pooled = pooled
            .as_ref()
            .map(|(entries, ratio, selection)| decode::Pooled {
                entries,
                ratio: *ratio,
                selection: *selection,
            });
        if !decode::takes(self.ctx(), heads, head_dim, &window_pool, pooled.as_ref()) {
            return None;
        }
        Some(decode::decode(
            self.ctx(),
            q,
            self.tensor(positions),
            self.tensor(request_of_token),
            floor.map(|floor| self.tensor(floor)),
            &window_pool,
            span,
            pooled,
            self.tensor(sink),
            heads,
            head_dim,
            sm_scale,
            turn,
            &mut self.tensor(o_out),
        ))
    }

    /// DeepSeek-V4's window and sinks at a prefill's width in the tcgen05
    /// pass (`attn::compressed::prefill::window_only`); `None` where the profile or
    /// the shape keeps it off.
    #[cfg(feature = "cuda")]
    fn dsv4_window_prefill(
        &mut self,
        name: &'static str,
        window: &poem_ir::Attn,
        sink: poem_ir::ValueId,
        o_out: poem_ir::ValueId,
    ) -> Option<Result<(), kernels_cuda::Error>> {
        use kernels_cuda::attn::compressed::prefill;
        let poem_ir::Attn::PrefillLse {
            q,
            cache,
            window: span,
            head_dim,
            sm_scale,
            ..
        } = *window
        else {
            return None;
        };
        let from = self.ctx().tuning().compressed_prefill_from;
        let mut out = self.tensor(o_out);
        if from == 0 || out.rows < from {
            return None;
        }
        let positions = self.input_of(&poem_ir::RuntimeInput::Positions)?;
        let requests = self.input_of(&poem_ir::RuntimeInput::Geometry {
            space: 0,
            kind: poem_ir::GeomKind::RequestOfToken,
        })?;
        let window_pool = self.pool_whole(cache)?;
        let q = self.tensor(q);
        let heads = q.width / head_dim.max(1);
        if !prefill::takes_window(heads, head_dim, span, &window_pool) {
            return None;
        }
        Some(prefill::window_only(
            self.ctx(),
            name,
            q,
            self.tensor(positions),
            self.tensor(requests),
            self.recorded(),
            &window_pool,
            self.tensor(sink),
            sm_scale,
            &mut out,
        ))
    }

    /// A log-sum-exp attention with its sinks: landed in the split's fold
    /// where the profile's warp fold takes them, else by the sink kernel
    /// after, as the pair ran.
    fn lse_sink(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::LseSink {
            attn, sink, o_out, ..
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        // DeepSeek-V4's window and sinks (its layers without a pool, its
        // draft head): the one launch at a decode's rows; over fp8 rows that
        // or a refusal, never the FA2 kernels reading them as bf16.
        if let poem_ir::Attn::PrefillLse { .. } = **attn {
            // A prefill's rows, where the profile says
            // (`kernel.sdpa.compressed_prefill_from`): the window in the tcgen05 pass.
            #[cfg(feature = "cuda")]
            if let Some(landed) = self.dsv4_window_prefill(op.name(), attn, *sink, *o_out) {
                return landed;
            }
            if self.fp8_rows(attn) {
                return self
                    .dsv4_decode(attn, None, *sink, *o_out, None)
                    .unwrap_or_else(|| Err(fp8_unread(op.name())));
            }
            let rows = self.tensor(*o_out).rows;
            if rows <= self.ctx().tuning().compressed_decode_until
                && let Some(landed) = self.dsv4_decode(attn, None, *sink, *o_out, None)
            {
                return landed;
            }
        }
        let sinks = self.tensor(*sink);
        match **attn {
            poem_ir::Attn::DecodeLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o: _,
                lse,
            } => kernels_cuda::attn::decode_lse_sink(
                self.ctx(),
                self.ragged_q(q),
                self.decode_plan(plan),
                &self.pool_absolute(cache),
                window,
                head_dim,
                sm_scale,
                &mut self.tensor(*o_out),
                &mut self.tensor(lse),
                sinks,
            ),
            poem_ir::Attn::PrefillLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o: _,
                lse,
                // (A floored window is no `lse_sink`'s: the rule takes none.)
                floor: _,
            } => {
                kernels_cuda::attn::prefill_lse_sink(
                    self.ctx(),
                    self.ragged_q(q),
                    self.prefill_plan(plan),
                    &self.pool_absolute(cache),
                    window,
                    head_dim,
                    kv_heads,
                    sm_scale,
                    &mut self.tensor(*o_out),
                    &mut self.tensor(lse),
                    sinks,
                )?;
                self.capture_scores(q, plan, cache, window, head_dim, kv_heads, sm_scale, lse)
            }
            _ => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
        }
    }

    /// Qwen3.5's attention prologue in one launch up to
    /// `kernel.elemwise.qg_prologue_until` rows on a bf16 NHD pool, else as traced.
    fn qg_qknorm_rope_append(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::{norm, norm_rope, rope};
        let Fused::QgQknormRopeAppend {
            qg,
            k,
            v,
            positions,
            q_norm,
            k_norm,
            head_dim,
            q_eps,
            k_eps,
            rotary_dim,
            theta,
            span,
            cache,
            write_page,
            write_offset,
            q_split,
            gate,
            q_normed,
            k_normed,
            q_out,
            k_out: _,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let pool = self.pool(*cache);
        let append = rope::YarnAppend {
            v: self.tensor(*v),
            pool: &pool,
            write_page: self.tensor(*write_page),
            write_offset: self.tensor(*write_offset),
        };
        if self.tensor(*q_out).rows <= self.ctx().tuning().qg_prologue_until
            && append.fits()
            && (*head_dim == 128 || *head_dim == 256)
            && span.is_multiple_of(*head_dim / 32)
        {
            return norm_rope::qg_qknorm_rope_append(
                self.ctx(),
                self.tensor(*qg),
                self.tensor(*k),
                self.tensor(*q_norm),
                self.tensor(*k_norm),
                self.tensor(*positions),
                *head_dim,
                *rotary_dim,
                *span,
                *theta,
                *q_eps,
                *k_eps,
                &mut self.tensor(*q_out),
                &mut self.tensor(*gate),
                append,
            );
        }
        kernels_cuda::gather::split_q_gate(
            self.ctx(),
            self.tensor(*qg),
            *head_dim,
            &mut self.tensor(*q_split),
            &mut self.tensor(*gate),
        )?;
        norm::rmsnorm_per_head_plus_one(
            self.ctx(),
            self.tensor(*q_split),
            self.tensor(*q_norm),
            *head_dim,
            *q_eps,
            &mut self.tensor(*q_normed),
        )?;
        norm::rmsnorm_per_head_plus_one(
            self.ctx(),
            self.tensor(*k),
            self.tensor(*k_norm),
            *head_dim,
            *k_eps,
            &mut self.tensor(*k_normed),
        )?;
        rope::partial(
            self.ctx(),
            &mut self.tensor(*q_normed),
            &mut self.tensor(*k_normed),
            self.tensor(*positions),
            *rotary_dim,
            *head_dim,
            *theta,
            *span,
        )?;
        kernels_cuda::attn::kv_append(
            self.ctx(),
            self.ragged_lanes(*k_normed),
            self.tensor(*v),
            &pool,
            self.tensor(*write_page),
            self.tensor(*write_offset),
        )
    }

    fn split_rope_yarn_append(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::rope;
        let Fused::SplitRopeYarnAppend {
            packed,
            q_width,
            kv_width,
            positions,
            head_dim,
            theta,
            factor,
            beta_fast,
            beta_slow,
            attention_factor,
            original_max_position,
            interleaved,
            cache,
            write_page,
            write_offset,
            q_out,
            k_out,
            v_out,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let r = rope::YarnRope {
            head_dim: *head_dim,
            theta: *theta,
            yarn: rope::Yarn {
                factor: *factor,
                beta_fast: *beta_fast,
                beta_slow: *beta_slow,
                original_max_position: *original_max_position,
            },
            attention_factor: *attention_factor,
            interleaved: *interleaved,
        };
        let pool = self.pool(*cache);
        let (mut qt, mut kt, mut vt) = (
            self.tensor(*q_out),
            self.tensor(*k_out),
            self.tensor(*v_out),
        );
        let append = rope::YarnAppend {
            v: vt,
            pool: &pool,
            write_page: self.tensor(*write_page),
            write_offset: self.tensor(*write_offset),
        };
        if append.fits() {
            return rope::yarn_split_append(
                self.ctx(),
                self.tensor(*packed),
                *kv_width,
                &mut qt,
                self.tensor(*positions),
                r,
                append,
            );
        }
        kernels_cuda::gather::split_qkv(
            self.ctx(),
            self.tensor(*packed),
            *q_width,
            *kv_width,
            &mut qt,
            &mut kt,
            &mut vt,
        )?;
        rope::yarn(self.ctx(), &mut qt, &mut kt, self.tensor(*positions), r)?;
        kernels_cuda::attn::kv_append(
            self.ctx(),
            self.ragged_lanes(*k_out),
            vt,
            &pool,
            self.tensor(*write_page),
            self.tensor(*write_offset),
        )
    }

    /// DeepSeek-V4's compressor at its boundaries in one launch where the profile
    /// says (`kernel.elemwise.rows_vec`) and the shape fits, else unfused.
    fn pool_compress(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::{norm_rope, rope};
        let Fused::PoolCompress {
            gather,
            weight,
            eps,
            latent,
            rotate,
            append,
            state: state_op,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        // The state written in the same launch where every request holds
        // one row of this fire (its boundary gathers only its own row of
        // it); else written first, as the trace had it.
        let one_a_request = {
            let window = self.window();
            !window.indptr_host.is_empty()
                && window
                    .indptr_host
                    .windows(2)
                    .all(|lane| lane[1] - lane[0] <= 1)
        };
        let state_rows = match state_op.as_deref() {
            Some(poem_ir::Attn::PoolStateWrite {
                kv,
                score,
                write_page,
                write_offset,
                ..
            }) if one_a_request => Some(kernels_cuda::attn::compressed::pool::StateRows {
                kv: self.tensor(*kv),
                score: self.tensor(*score),
                write_page: self.tensor(*write_page),
                write_offset: self.tensor(*write_offset),
            }),
            _ => None,
        };
        let state_rides = state_rows.is_some();
        if !state_rides && let Some(state) = state_op {
            self.attention(state)?;
        }
        let (
            poem_ir::Attn::PoolGather {
                boundary_pos,
                boundary_req,
                pages,
                ape,
                head_dim,
                ratio,
                entries,
            },
            poem_ir::Elemwise::RopePartialLast {
                positions,
                rotary_dim,
                theta,
                interleaved,
                inverse,
                yarn,
                ..
            },
            poem_ir::Attn::PoolKvAppend { pool: into, .. },
        ) = (&**gather, &**rotate, &**append)
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let yarn = yarn.map(|y| rope::Yarn {
            factor: y.factor,
            beta_fast: y.beta_fast,
            beta_slow: y.beta_slow,
            original_max_position: y.original_max_position,
        });
        let slabs = self.slabs(*pages);
        let state = self.pool(*pages);
        let out = self.pool(*into);
        let gather_ape = ape.map_or(slabs.ape, |id| self.tensor(id));
        let landed = kernels_cuda::attn::compressed::pool::compress(
            self.ctx(),
            self.tensor(*boundary_pos),
            self.tensor(*boundary_req),
            &state,
            *head_dim,
            *ratio,
            slabs.state_kv,
            slabs.state_score,
            gather_ape,
            self.tensor(*weight),
            *eps,
            self.tensor(*positions),
            *rotary_dim,
            *theta,
            *interleaved && !*inverse,
            yarn,
            &out,
            state_rows,
        )?;
        if landed {
            return Ok(());
        }
        if state_rides && let Some(state) = state_op {
            self.attention(state)?;
        }
        self.attention(gather)?;
        norm_rope::norm_rope_last(
            self.ctx(),
            self.tensor(*entries),
            Some(self.tensor(*weight)),
            *head_dim,
            *eps,
            self.tensor(*positions),
            *rotary_dim,
            *theta,
            *interleaved,
            *inverse,
            yarn,
            &mut self.tensor(*latent),
        )?;
        self.attention(append)
    }

    /// An index's queries turned as the ranking loads them where its scorer
    /// can (the tensor-core one, interleaved forward turns); else the turn,
    /// then the ranking, as the unfused ops fire them.
    fn index_topk_rope(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::IndexTopkRope { rotate, rank } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        if let (
            poem_ir::Elemwise::RopePartialLast {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                interleaved: true,
                inverse: false,
                yarn,
                ..
            },
            poem_ir::Attn::IndexTopk {
                weights,
                keys,
                heads,
                head_dim: rank_dim,
                top_k,
                ratio,
                score: poem_ir::IndexScore::Pooled,
                selection,
                ..
            },
        ) = (&**rotate, &**rank)
            && head_dim == rank_dim
        {
            let turn = kernels_cuda::attn::index::QueryTurn {
                positions: self.tensor(*positions),
                dim: *rotary_dim,
                theta: *theta,
                yarn: yarn.map(|y| kernels_cuda::elemwise::rope::Yarn {
                    factor: y.factor,
                    beta_fast: y.beta_fast,
                    beta_slow: y.beta_slow,
                    original_max_position: y.original_max_position,
                }),
            };
            let landed = kernels_cuda::attn::index::topk_turned(
                self.ctx(),
                self.ragged_lanes(*q),
                weights.map(|w| self.tensor(w)),
                &self.pool(*keys),
                *heads,
                *head_dim,
                *top_k,
                *ratio,
                turn,
                &mut self.tensor(*selection),
            )?;
            if landed {
                return Ok(());
            }
        }
        self.elemwise(rotate)?;
        self.attention(rank)
    }

    /// DeepSeek-V4's q and kv planes normed and rotated in one launch (kv written
    /// as its fp8 row there, or after to a bf16 pool) under `kernel.elemwise.rows_vec`
    /// where the shape fits; else each as its own fused op.
    fn norm_rope_pair(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::{norm_rope, rope};
        let Fused::NormRopePair { q, kv } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let (
            Fused::NormRopeLast {
                x: qx,
                weight: None,
                head_dim: q_dim,
                eps: q_eps,
                positions,
                rotary_dim,
                theta,
                interleaved: q_interleaved,
                inverse: q_inverse,
                yarn: q_yarn,
                y: qy,
                ..
            },
            Fused::NormRopeLastAppend {
                x: kx,
                weight,
                head_dim: kv_dim,
                eps: kv_eps,
                positions: kv_positions,
                rotary_dim: kv_rotary,
                theta: kv_theta,
                interleaved: kv_interleaved,
                inverse: kv_inverse,
                yarn: kv_yarn,
                y: ky,
                cache,
                write_page,
                write_offset,
            },
        ) = (&**q, &**kv)
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let alike = positions == kv_positions
            && rotary_dim == kv_rotary
            && theta.to_bits() == kv_theta.to_bits()
            && *q_interleaved
            && *kv_interleaved
            && !*q_inverse
            && !*kv_inverse
            && q_yarn == kv_yarn;
        if alike {
            let pool = self.pool(*cache);
            let fp8 = kernels_cuda::attn::compressed::decode::fp8_ds(&pool);
            let yarn = q_yarn.map(|y| rope::Yarn {
                factor: y.factor,
                beta_fast: y.beta_fast,
                beta_slow: y.beta_slow,
                original_max_position: y.original_max_position,
            });
            let landed = norm_rope::norm_rope_pair(
                self.ctx(),
                norm_rope::Side {
                    x: self.tensor(*qx),
                    head_dim: *q_dim,
                    eps: *q_eps,
                    y: &mut self.tensor(*qy),
                },
                norm_rope::Side {
                    x: self.tensor(*kx),
                    head_dim: *kv_dim,
                    eps: *kv_eps,
                    y: &mut self.tensor(*ky),
                },
                self.tensor(*weight),
                self.tensor(*positions),
                *rotary_dim,
                *theta,
                yarn,
                fp8.then_some(norm_rope::Append {
                    pool: &pool,
                    write_page: self.tensor(*write_page),
                    write_offset: self.tensor(*write_offset),
                }),
            )?;
            if landed {
                if fp8 {
                    return Ok(());
                }
                if kernels_cuda::attn::compressed::decode::ds_rows(&pool) {
                    // V4.1's rows: the write quantizes them apart.
                    return kernels_cuda::attn::compressed::decode::append(
                        self.ctx(),
                        self.ragged_lanes(*ky).data,
                        &pool,
                        self.tensor(*write_page),
                        self.tensor(*write_offset),
                    );
                }
                return kernels_cuda::attn::kv_append_shared(
                    self.ctx(),
                    self.ragged_lanes(*ky),
                    &pool,
                    self.tensor(*write_page),
                    self.tensor(*write_offset),
                );
            }
        }
        self.fused_elemwise(q)?;
        self.norm_rope_last_append(kv)
    }

    /// DeepSeek-V4's kv plane normed, rotated and written in one launch into a pool
    /// of DeepSeek's fp8 rows (under `kernel.elemwise.rows_vec`); else unfused.
    fn norm_rope_last_append(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::{norm_rope, rope};
        let Fused::NormRopeLastAppend {
            x,
            weight,
            head_dim,
            eps,
            positions,
            rotary_dim,
            theta,
            interleaved,
            inverse,
            yarn,
            y,
            cache,
            write_page,
            write_offset,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let pool = self.pool(*cache);
        let yarn = yarn.map(|y| rope::Yarn {
            factor: y.factor,
            beta_fast: y.beta_fast,
            beta_slow: y.beta_slow,
            original_max_position: y.original_max_position,
        });
        let fp8 = kernels_cuda::attn::compressed::decode::fp8_ds(&pool);
        let landed = norm_rope::norm_rope_last_append(
            self.ctx(),
            self.tensor(*x),
            self.tensor(*weight),
            *head_dim,
            *eps,
            self.tensor(*positions),
            *rotary_dim,
            *theta,
            *interleaved,
            *inverse,
            yarn,
            &mut self.tensor(*y),
            fp8.then_some(norm_rope::Append {
                pool: &pool,
                write_page: self.tensor(*write_page),
                write_offset: self.tensor(*write_offset),
            }),
        )?;
        if landed {
            return Ok(());
        }
        norm_rope::norm_rope_last(
            self.ctx(),
            self.tensor(*x),
            Some(self.tensor(*weight)),
            *head_dim,
            *eps,
            self.tensor(*positions),
            *rotary_dim,
            *theta,
            *interleaved,
            *inverse,
            yarn,
            &mut self.tensor(*y),
        )?;
        if kernels_cuda::attn::compressed::decode::ds_rows(&pool) {
            kernels_cuda::attn::compressed::decode::append(
                self.ctx(),
                self.ragged_lanes(*y).data,
                &pool,
                self.tensor(*write_page),
                self.tensor(*write_offset),
            )
        } else {
            kernels_cuda::attn::kv_append_shared(
                self.ctx(),
                self.ragged_lanes(*y),
                &pool,
                self.tensor(*write_page),
                self.tensor(*write_offset),
            )
        }
    }

    /// YaRN and the cache write in one launch where the pool is the plain
    /// bf16 one `kv_append` writes directly; else the two as they were.
    fn rope_yarn_append(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::rope;
        let Fused::RopeYarnAppend {
            q,
            k,
            v,
            positions,
            head_dim,
            theta,
            factor,
            beta_fast,
            beta_slow,
            attention_factor,
            original_max_position,
            interleaved,
            cache,
            write_page,
            write_offset,
            ..
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let r = rope::YarnRope {
            head_dim: *head_dim,
            theta: *theta,
            yarn: rope::Yarn {
                factor: *factor,
                beta_fast: *beta_fast,
                beta_slow: *beta_slow,
                original_max_position: *original_max_position,
            },
            attention_factor: *attention_factor,
            interleaved: *interleaved,
        };
        let pool = self.pool(*cache);
        let append = rope::YarnAppend {
            v: self.tensor(*v),
            pool: &pool,
            write_page: self.tensor(*write_page),
            write_offset: self.tensor(*write_offset),
        };
        let (mut qt, mut kt) = (self.tensor(*q), self.tensor(*k));
        if append.fits() {
            return rope::yarn_append(
                self.ctx(),
                &mut qt,
                &mut kt,
                self.tensor(*positions),
                r,
                append,
            );
        }
        rope::yarn(self.ctx(), &mut qt, &mut kt, self.tensor(*positions), r)?;
        kernels_cuda::attn::kv_append(
            self.ctx(),
            self.ragged_lanes(*k),
            self.tensor(*v),
            &pool,
            self.tensor(*write_page),
            self.tensor(*write_offset),
        )
    }

    /// The fused qkv write; `rest`, a row split's second half it lands too.
    fn custom_cuda(
        &mut self,
        op: &Fused,
        rest: Option<kernels_cuda::Tensor>,
    ) -> Result<(), kernels_cuda::Error> {
        match op {
            Fused::QkvFusedQknormRopeVnormWrite {
                packed,
                positions,
                q_norm_weight,
                k_norm_weight,
                v_norm,
                eps,
                cache,
                write_page,
                write_offset,
                kv_heads,
                head_dim,
                theta,
                rotary_dim,
                span,
                qk_norm,
                v_scale,
                widths,
                q,
            } => qkv_append::qkv_fused_qknorm_rope_vnorm_write(
                self.ctx(),
                self.tensor(*packed),
                positions.map(|p| self.tensor(p)),
                q_norm_weight.map(|w| self.tensor(w)),
                k_norm_weight.map(|w| self.tensor(w)),
                *v_norm,
                *eps,
                &self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
                *kv_heads,
                *head_dim,
                *theta,
                *rotary_dim,
                *span,
                *qk_norm,
                *v_scale,
                *widths,
                rest,
                &mut self.tensor(*q),
            ),
            _ => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
        }
    }
}

/// A read of DeepSeek's fp8 rows nothing here takes: refused, as no other
/// reader knows the plane.
fn fp8_unread(op: &'static str) -> kernels_cuda::Error {
    kernels_cuda::Error::Backend {
        op,
        detail: "the cache holds DeepSeek's fp8 rows and only `attn::compressed::decode::decode` reads them; \
                 it does not take this shape or this region's gathered seats"
            .to_string(),
    }
}
