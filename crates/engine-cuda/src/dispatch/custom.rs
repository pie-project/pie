use kernels_cuda::custom;
use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        match op {
            Fused::QkvFusedQknormRopeVnormWrite { .. } => self.custom_cuda(op),
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
            | Fused::NormRopeLast { .. }
            | Fused::NormModulate { .. }
            | Fused::GatedResidualNormModulate { .. } => {
                let mut outputs = Vec::new();
                op.outputs(&mut outputs);
                if outputs.first().is_some_and(|out| self.lane_shaped(*out)) {
                    self.unseated(|| self.fused_elementwise(op))
                } else {
                    self.fused_elementwise(op)
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

    /// A sublayer's whole pre-mix: in two launches where the profile's tiled
    /// hyper-connection kernels run (the normed streams, the mixes and the
    /// pre-mixed row left unwritten, as nothing reads them), else the four as
    /// the trace had them.
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
            && let poem_ir::Elementwise::HcRmsnormF32 { streams, eps, .. } = &**norm
            && let poem_ir::Elementwise::HcProject {
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
            self.elementwise(gates)?;
            return self.elementwise(rmsnorm);
        }
        if self.ctx().tuning().hc_tile
            && let (
                poem_ir::Elementwise::HcRmsnormF32 {
                    streams,
                    eps: mix_eps,
                    ..
                },
                poem_ir::Elementwise::HcProject {
                    weight: plane,
                    stream_count,
                    mixes,
                    ..
                },
                poem_ir::Elementwise::HcGates {
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
                poem_ir::Elementwise::Rmsnorm { weight, eps, y, .. },
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
        self.elementwise(norm)?;
        self.elementwise(project)?;
        self.elementwise(gates)?;
        self.elementwise(rmsnorm)
    }

    /// The combined gates and the RMS norm over their pre-mixed row: in one
    /// launch where the profile's tiled hyper-connection kernels run (the
    /// pre-mixed row left unwritten, as nothing reads it), else the two as
    /// the trace had them.
    /// AttnRes's blend and the norm of its row: in one pass where the
    /// profile says (`kernel.attn_res.fused_norm`) and the row takes it,
    /// else the two.
    fn res_blend_rmsnorm(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::ResBlendRmsnorm { blend, norm } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        if let (
            poem_ir::Elementwise::ResBlend {
                prefix,
                blocks,
                weight: blend_weight,
                eps: blend_eps,
                proj,
                ..
            },
            poem_ir::Elementwise::Rmsnorm { weight, eps, y, .. },
        ) = (&**blend, &**norm)
            && self.ctx().tuning().attn_res_fused_norm
        {
            let blocks: Vec<kernels_cuda::tensor::Tensor> =
                blocks.iter().map(|b| self.tensor(*b)).collect();
            if kernels_cuda::elemwise::norm::res_blend_norm(
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
        self.elementwise(blend)?;
        self.elementwise(norm)
    }

    fn hc_gates_norm(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::HcGatesNorm { gates, norm } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let (
            poem_ir::Elementwise::HcGates {
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
            poem_ir::Elementwise::Rmsnorm { weight, eps, y, .. },
        ) = (&**gates, &**norm)
        else {
            self.elementwise(gates)?;
            return self.elementwise(norm);
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
        self.elementwise(gates)?;
        self.elementwise(norm)
    }

    /// A stream norm and the mix projection over it: in one pass over the
    /// streams where the profile's tiled hyper-connection kernels run (the
    /// normed rows left unwritten, as nothing reads them), else the two as
    /// the trace had them.
    fn hc_prenorm_project(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        let Fused::HcPrenormProject { norm, project } = op else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let (
            poem_ir::Elementwise::HcRmsnormF32 { streams, eps, .. },
            poem_ir::Elementwise::HcProject {
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
        self.elementwise(norm)?;
        self.elementwise(project)
    }

    /// A latent attention's prefill: expanded (a key and value head a query
    /// head, as vLLM's MLA prefill runs it) from `kernel.mla.expanded_from`
    /// rows where the device and the heads take it, else the split, the
    /// absorbs and the scoring as the trace had them.
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
            poem_ir::Attention::MlaAbsorbQ {
                q_nope,
                kv_b,
                heads,
                kv_lora_rank,
                nope_dim,
                v_head_dim,
                ..
            },
            poem_ir::Attention::MlaPrefill {
                q,
                q_pe,
                cache,
                window: None,
                sm_scale,
                lse: None,
                ..
            },
            poem_ir::Attention::MlaAbsorbOut { o, .. },
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
                    Some(poem_ir::Attention::MlaSplitQB { q_b, .. }) => {
                        ExpandedQ::Packed(self.tensor(*q_b))
                    }
                    _ => ExpandedQ::Split {
                        q_nope: self.tensor(*q_nope),
                        q_pe: q_pe_t,
                    },
                };
                let kv_b = self.dense_or_decoded(
                    "attention.mla_prefill_expanded",
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
            self.elementwise(rotate)?;
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
            sink,
            head_dim,
            o_out,
            rotate,
        } = op
        else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        let poem_ir::Attention::MergeLse { o, lse, .. } = **merge else {
            return Err(kernels_cuda::Error::Unsupported { op: op.name() });
        };
        // The output's tail turned back with the output where the turn is
        // DeepSeek-V4's (interleaved, inverse, at the pool's positions).
        let turn = match (rotate.as_deref(), &**pool) {
            (
                Some(poem_ir::Elementwise::RopePartialLast {
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
                poem_ir::Attention::PoolLse { positions, .. }
                | poem_ir::Attention::PoolLseSelected { positions, .. },
            ) if q == o_out && at == positions && d == head_dim => {
                Some(kernels_cuda::attn::dsv4::Turn {
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
        // A decode's rows, up to what the profile says
        // (`kernel.sdpa.dsv4_decode_until`): the four in one launch.
        let rows = self.tensor(*o_out).rows;
        if rows <= self.ctx().tuning().dsv4_decode_until
            && let Some(landed) = self.dsv4_decode(window, Some(pool), *sink, *o_out, turn)
        {
            return landed.map(|()| turn.is_some());
        }
        // A prefill's rows, where the profile says
        // (`kernel.sdpa.dsv4_prefill_from`): the four in one tcgen05 pass.
        #[cfg(feature = "cuda")]
        if let (
            poem_ir::Attention::PrefillLse {
                q,
                cache,
                window: span,
                head_dim: d,
                sm_scale,
                ..
            },
            poem_ir::Attention::PoolLse {
                positions,
                request_of_token,
                entries,
                ratio,
                heads,
                ..
            }
            | poem_ir::Attention::PoolLseSelected {
                positions,
                request_of_token,
                entries,
                ratio,
                heads,
                ..
            },
        ) = (&**window, &**pool)
        {
            use kernels_cuda::attn::dsv4_prefill::{self, Pooled};
            let from = self.ctx().tuning().dsv4_prefill_from;
            let mut out = self.tensor(*o_out);
            // Its request ids index the whole fire's page tables too.
            if from > 0
                && out.rows >= from
                && let (Some(window_pool), Some(entries)) =
                    (self.pool_whole(*cache), self.pool_whole(*entries))
            {
                let selection = match &**pool {
                    poem_ir::Attention::PoolLseSelected {
                        selection, top_k, ..
                    } => Some((self.tensor(*selection), *top_k)),
                    _ => None,
                };
                let pooled = Pooled {
                    entries: &entries,
                    ratio: *ratio,
                    selection,
                };
                if dsv4_prefill::takes(*heads, *d, *span, &window_pool, &pooled) {
                    return dsv4_prefill::prefill(
                        self.ctx(),
                        op.name(),
                        self.tensor(*q),
                        self.tensor(*positions),
                        self.tensor(*request_of_token),
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
        // DeepSeek's fp8 rows have no other reader: the one launch at any
        // width, or a refusal, never the four ops reading them as bf16.
        if self.fp8_rows(window) {
            return self
                .dsv4_decode(window, Some(pool), *sink, *o_out, turn)
                .unwrap_or_else(|| Err(fp8_unread(op.name())))
                .map(|()| turn.is_some());
        }
        self.attention(window)?;
        self.attention(pool)?;
        self.attention(merge)?;
        self.attention(&poem_ir::Attention::Sink {
            o,
            lse,
            sink: *sink,
            head_dim: *head_dim,
            o_out: *o_out,
        })
        .map(|()| false)
    }

    /// Whether `window`'s cache holds DeepSeek's fp8 rows.
    fn fp8_rows(&self, window: &poem_ir::Attention) -> bool {
        match *window {
            poem_ir::Attention::PrefillLse { cache, .. } => {
                kernels_cuda::attn::dsv4::fp8_ds(&self.pool(cache))
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

    /// DeepSeek-V4's window (`window`, a `PrefillLse`), its pool if any
    /// (`PoolLse` or `PoolLseSelected`) and the sinks in one launch
    /// (`attn::dsv4::decode`), the sinked output in `o_out`; `None` where it
    /// does not take the shape or the region reads gathered seats.
    fn dsv4_decode(
        &mut self,
        window: &poem_ir::Attention,
        pool: Option<&poem_ir::Attention>,
        sink: poem_ir::ValueId,
        o_out: poem_ir::ValueId,
        turn: Option<kernels_cuda::attn::dsv4::Turn>,
    ) -> Option<Result<(), kernels_cuda::Error>> {
        use kernels_cuda::attn::dsv4;
        let poem_ir::Attention::PrefillLse {
            q,
            cache,
            window: Some(span),
            head_dim,
            sm_scale,
            ..
        } = *window
        else {
            return None;
        };
        let (positions, request_of_token, pooled) = match pool {
            Some(
                poem_ir::Attention::PoolLse {
                    positions,
                    request_of_token,
                    entries,
                    ratio,
                    ..
                }
                | poem_ir::Attention::PoolLseSelected {
                    positions,
                    request_of_token,
                    entries,
                    ratio,
                    ..
                },
            ) => {
                let selection = match pool {
                    Some(poem_ir::Attention::PoolLseSelected {
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
            .map(|(entries, ratio, selection)| dsv4::Pooled {
                entries,
                ratio: *ratio,
                selection: *selection,
            });
        if !dsv4::takes(heads, head_dim, &window_pool, pooled.as_ref()) {
            return None;
        }
        Some(dsv4::decode(
            self.ctx(),
            q,
            self.tensor(positions),
            self.tensor(request_of_token),
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
        // or a refusal, never FlashInfer reading them as bf16.
        if let poem_ir::Attention::PrefillLse { .. } = **attn {
            if self.fp8_rows(attn) {
                return self
                    .dsv4_decode(attn, None, *sink, *o_out, None)
                    .unwrap_or_else(|| Err(fp8_unread(op.name())));
            }
            let rows = self.tensor(*o_out).rows;
            if rows <= self.ctx().tuning().dsv4_decode_until
                && let Some(landed) = self.dsv4_decode(attn, None, *sink, *o_out, None)
            {
                return landed;
            }
        }
        let sinks = self.tensor(*sink);
        match **attn {
            poem_ir::Attention::DecodeLse {
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
            poem_ir::Attention::PrefillLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o: _,
                lse,
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

    /// The packed row's q, k and v rotated and written in one launch where
    /// the pool is the plain bf16 one; else split, then YaRN and the write as
    /// `rope_yarn_append` runs them.
    /// Qwen3.5's attention prologue: one launch up to the rows the profile
    /// says (`kernel.elemwise.qg_prologue_until`) where the cache is a bf16
    /// NHD pool, else the cut, the two norms, the rotation and the append as
    /// the trace had them.
    fn qg_qknorm_rope_append(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::{norm, rope};
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
            return rope::qg_qknorm_rope_append(
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
        kernels_cuda::layout::split_q_gate(
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
                *head_dim,
                *theta,
                *factor,
                *beta_fast,
                *beta_slow,
                *attention_factor,
                *original_max_position,
                *interleaved,
                append,
            );
        }
        kernels_cuda::layout::split_qkv(
            self.ctx(),
            self.tensor(*packed),
            *q_width,
            *kv_width,
            &mut qt,
            &mut kt,
            &mut vt,
        )?;
        rope::yarn(
            self.ctx(),
            &mut qt,
            &mut kt,
            self.tensor(*positions),
            *head_dim,
            *theta,
            *factor,
            *beta_fast,
            *beta_slow,
            *attention_factor,
            *original_max_position,
            *interleaved,
        )?;
        kernels_cuda::attn::kv_append(
            self.ctx(),
            self.ragged_lanes(*k_out),
            vt,
            &pool,
            self.tensor(*write_page),
            self.tensor(*write_offset),
        )
    }

    /// DeepSeek-V4's compressor at its boundaries in one launch where the
    /// profile says (`kernel.elemwise.rows_vec`) and the shape fits; else the
    /// gather, the norm and rotation (`norm_rope_last`), and the append, as
    /// the unfused ops fire them.
    fn pool_compress(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::rope;
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
            Some(poem_ir::Attention::PoolStateWrite {
                kv,
                score,
                write_page,
                write_offset,
                ..
            }) if one_a_request => Some(kernels_cuda::attn::pool::StateRows {
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
            poem_ir::Attention::PoolGather {
                boundary_pos,
                boundary_req,
                pages,
                ape,
                head_dim,
                ratio,
                entries,
            },
            poem_ir::Elementwise::RopePartialLast {
                positions,
                rotary_dim,
                theta,
                interleaved,
                inverse,
                yarn,
                ..
            },
            poem_ir::Attention::PoolKvAppend { pool: into, .. },
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
        let landed = kernels_cuda::attn::pool::compress(
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
        rope::norm_rope_last(
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
            poem_ir::Elementwise::RopePartialLast {
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
            poem_ir::Attention::IndexTopk {
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
        self.elementwise(rotate)?;
        self.attention(rank)
    }

    /// DeepSeek-V4's q and kv plane normed and rotated in one launch (the
    /// plane written as its fp8 row there too, or after it to a bf16 pool)
    /// where the profile says (`kernel.elemwise.rows_vec`) and the shape
    /// fits; else each as its own fused op.
    fn norm_rope_pair(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::rope;
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
            let fp8 = kernels_cuda::attn::dsv4::fp8_ds(&pool);
            let yarn = q_yarn.map(|y| rope::Yarn {
                factor: y.factor,
                beta_fast: y.beta_fast,
                beta_slow: y.beta_slow,
                original_max_position: y.original_max_position,
            });
            let landed = rope::norm_rope_pair(
                self.ctx(),
                rope::Side {
                    x: self.tensor(*qx),
                    head_dim: *q_dim,
                    eps: *q_eps,
                    y: &mut self.tensor(*qy),
                },
                rope::Side {
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
                fp8.then_some(rope::Append {
                    pool: &pool,
                    write_page: self.tensor(*write_page),
                    write_offset: self.tensor(*write_offset),
                }),
            )?;
            if landed {
                if fp8 {
                    return Ok(());
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
        self.fused_elementwise(q)?;
        self.norm_rope_last_append(kv)
    }

    /// DeepSeek-V4's kv plane normed, rotated and written in one launch
    /// where the pool holds DeepSeek's fp8 rows (and the profile says,
    /// `kernel.elemwise.rows_vec`); else the norm and rotation, then the
    /// write, as the unfused ops fire them.
    fn norm_rope_last_append(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        use kernels_cuda::elemwise::rope;
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
        let fp8 = kernels_cuda::attn::dsv4::fp8_ds(&pool);
        let landed = rope::norm_rope_last_append(
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
            fp8.then_some(rope::Append {
                pool: &pool,
                write_page: self.tensor(*write_page),
                write_offset: self.tensor(*write_offset),
            }),
        )?;
        if landed {
            return Ok(());
        }
        rope::norm_rope_last(
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
        if fp8 {
            kernels_cuda::attn::dsv4::append(
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
                *head_dim,
                *theta,
                *factor,
                *beta_fast,
                *beta_slow,
                *attention_factor,
                *original_max_position,
                *interleaved,
                append,
            );
        }
        rope::yarn(
            self.ctx(),
            &mut qt,
            &mut kt,
            self.tensor(*positions),
            *head_dim,
            *theta,
            *factor,
            *beta_fast,
            *beta_slow,
            *attention_factor,
            *original_max_position,
            *interleaved,
        )?;
        kernels_cuda::attn::kv_append(
            self.ctx(),
            self.ragged_lanes(*k),
            self.tensor(*v),
            &pool,
            self.tensor(*write_page),
            self.tensor(*write_offset),
        )
    }

    fn custom_cuda(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        match op {
            Fused::QkvFusedQknormRopeVnormWrite {
                packed,
                positions,
                q_norm_weight,
                k_norm_weight,
                eps,
                cache,
                write_page,
                write_offset,
                kv_heads,
                head_dim,
                theta,
                rotary_dim,
                q,
            } => custom::qkv_fused_qknorm_rope_vnorm_write(
                self.ctx(),
                self.tensor(*packed),
                self.tensor(*positions),
                self.tensor(*q_norm_weight),
                self.tensor(*k_norm_weight),
                *eps,
                &self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
                *kv_heads,
                *head_dim,
                *theta,
                *rotary_dim,
                &mut self.tensor(*q),
            ),
            _ => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
        }
    }
}

impl poem_exec::DispatchProbe for Run<'_> {
    fn probe(&mut self, node: &poem_ir::Node) {
        use poem_ir::Operands;
        let tag = crate::record::PTR_TAG.load(std::sync::atomic::Ordering::Relaxed);
        if tag != 0 {
            let mut ins = Vec::new();
            let mut outs = Vec::new();
            node.op.inputs(&mut ins);
            node.op.outputs(&mut outs);
            let show = |run: &Self, ids: &[poem_ir::ValueId]| -> String {
                ids.iter()
                    .map(|id| {
                        let Some(decl) = run.values().get(id.0 as usize) else {
                            return format!("{}:?", id.0);
                        };
                        match &decl.def {
                            poem_ir::Def::Op(_)
                            | poem_ir::Def::Merge(_)
                            | poem_ir::Def::Weight(_)
                                if run.resolvable(*id) =>
                            {
                                let t = run.tensor(*id);
                                format!("{}:{:#x}/{}x{}", id.0, t.ptr, t.rows, t.width)
                            }
                            poem_ir::Def::Op(_) | poem_ir::Def::Merge(_) => {
                                format!("{}:unslotted", id.0)
                            }
                            poem_ir::Def::Weight(_) => format!("{}:planes", id.0),
                            poem_ir::Def::Input(kind) => {
                                let t = run.tensor(*id);
                                let short = format!("{kind:?}").replace(' ', "");
                                format!(
                                    "{}:{:#x}/{}x{}@{}",
                                    id.0,
                                    t.ptr,
                                    t.rows,
                                    t.width,
                                    short.chars().take(28).collect::<String>()
                                )
                            }
                            other => format!(
                                "{}:{}",
                                id.0,
                                format!("{other:?}").chars().take(12).collect::<String>()
                            ),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            eprintln!(
                "[ptr-trace] tag={tag} {} layer={:?} in=[{}] out=[{}]",
                node.op.name(),
                node.layer,
                show(self, &ins),
                show(self, &outs)
            );
        }
        if !crate::serve::diag::on().nan_check {
            return;
        }
        let mut outs = Vec::new();
        node.op.outputs(&mut outs);
        for id in outs {
            let Some(decl) = self.values().get(id.0 as usize) else {
                continue;
            };
            let poem_ir::Ty::Tensor { dtype, .. } = &decl.ty else {
                continue;
            };
            let elem: usize = match dtype {
                poem_ir::Dtype::Bf16 | poem_ir::Dtype::F16 => 2,
                poem_ir::Dtype::F32 => 4,
                _ => continue,
            };
            let t = self.tensor(id);
            let total = t.rows as usize * t.width as usize * elem;
            if total == 0 || t.ptr == 0 {
                continue;
            }
            let bytes = total.min(1 << 20);
            let mut host = vec![0u8; bytes];
            if crate::device::copy_any(self.ctx().stream(), host.as_mut_ptr() as u64, t.ptr, bytes)
                .is_err()
            {
                continue;
            }
            let bad = match dtype {
                poem_ir::Dtype::F32 => host
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .position(|c| !f32::from_le_bytes(*c).is_finite()),
                poem_ir::Dtype::Bf16 => host
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .position(|c| (u16::from_le_bytes(*c) & 0x7f80) == 0x7f80),
                _ => host
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .position(|c| (u16::from_le_bytes(*c) & 0x7c00) == 0x7c00),
            };
            if let Some(at) = bad {
                eprintln!(
                    "nan-check: {} layer={:?} value={} dtype={:?} rows={} width={} first non-finite at element {at}",
                    node.op.name(),
                    node.layer,
                    id.0,
                    dtype,
                    t.rows,
                    t.width
                );
            }
        }
    }
}

/// A read of DeepSeek's fp8 rows nothing here takes: refused, as no other
/// reader knows the plane.
fn fp8_unread(op: &'static str) -> kernels_cuda::Error {
    kernels_cuda::Error::Backend {
        op,
        detail: "the cache holds DeepSeek's fp8 rows and only `attn::dsv4::decode` reads them; \
                 it does not take this shape or this region's gathered seats"
            .to_string(),
    }
}
