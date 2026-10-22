use kernels_cuda::attn::qkv_append;
use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};

use crate::run::{HeldGate, HeldShared, Run};
use poem_ir::ValueId;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        match op {
            Fused::GdnStep { .. } => return self.gdn_step(op),
            Fused::QGateNormRopeAppend { .. } => return self.q_gate_append(op),
            Fused::QkvFusedQknormRopeVnormWrite { .. } => self.custom_cuda(op),
            Fused::MatmulGeglu { .. }
            | Fused::MlpSwiglu { .. }
            | Fused::LmHeadSoftcap { .. }
            | Fused::MatmulBias { .. } => self.fused_linear(op),
            // Kernels this backend does not ship as one: never formed for it,
            // and run as the primitives they stand for where a trace states
            // them.
            Fused::MatmulSwiglu { .. }
            | Fused::MatmulSplitRows { .. }
            | Fused::MoeSelectGlu { .. }
            | Fused::MoeExperts { .. }
            | Fused::LseSink { .. }
            | Fused::RopeYarnAppend { .. }
            | Fused::SplitRopeYarnAppend { .. } => return self.primitives(op),
            // CUDA's DeepSeek, hyper-connection and Qwen prologue kernels:
            // never formed here.
            Fused::NormRopeLast { .. }
            | Fused::NormRopeLastAppend { .. }
            | Fused::NormRopePair { .. }
            | Fused::IndexTopkRope { .. }
            | Fused::PoolCompress { .. }
            | Fused::PoolAttention { .. }
            | Fused::MlaPrefillExpanded { .. }
            | Fused::HcPre { .. }
            | Fused::HcGatesNorm { .. }
            | Fused::HcPrenormProject { .. }
            | Fused::ResBlendRmsnorm { .. }
            | Fused::QgQknormRopeAppend { .. }
            | Fused::SplitQkvFused { .. } => {
                Err(kernels_cuda::Error::Unsupported { op: op.name() })
            }
            Fused::ResidualAddRmsnorm { .. }
            | Fused::RmsnormResidualAdd { .. }
            | Fused::EmbedScaleAdd { .. }
            | Fused::EmbedScaleAddSelect { .. }
            | Fused::RmsnormRopePartialQ { .. }
            | Fused::HcNormProject { .. }
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
    /// A routed swiglu MLP over dense bf16 banks (selects through the plain
    /// banks, a swiglu or its clamp, the weighted sum) as the routed MLP
    /// `linear::moe` takes, with the value it sums into; `None` where the legs
    /// stay the caller's.
    fn routed_mlp(
        &mut self,
        select: &poem_ir::Linear,
        glu: &poem_ir::Linear,
        down: &poem_ir::Linear,
        sum: &poem_ir::Linear,
    ) -> Option<(kernels_cuda::linear::moe::RoutedMlp, ValueId)> {
        use poem_ir::Linear;
        let (
            Linear::MoeMatmulSelect {
                x,
                bank: gate_up,
                routes,
                y: packed,
            },
            Linear::MoeMatmulSelect {
                x: h,
                bank: down_bank,
                routes: down_routes,
                y: routed,
            },
            Linear::MoeWeightedSum {
                routed: summed,
                weights,
                y,
            },
        ) = (select, down, sum)
        else {
            return None;
        };
        let (intermediate, limit, glu_in, glu_out) = match glu {
            Linear::MlpSwiglu {
                packed,
                intermediate,
                y,
            } => (*intermediate, None, *packed, *y),
            Linear::MlpSwigluClamp {
                packed,
                intermediate,
                limit,
                y,
            } => (*intermediate, Some(*limit), *packed, *y),
            _ => return None,
        };
        if routes != down_routes || glu_in != *packed || glu_out != *h || summed != routed {
            return None;
        }
        // Banks in the matrix core's order run as their legs.
        if self.ordered_bank(*gate_up).is_some() || self.ordered_bank(*down_bank).is_some() {
            return None;
        }
        let (gate_up, gate_up_experts) = self.expert_bank(*gate_up);
        let (down, down_experts) = self.expert_bank(*down_bank);
        let p = kernels_cuda::linear::moe::RoutedMlp {
            x: self.tensor(*x),
            routes: self.tensor(*routes),
            weights: self.tensor(*weights),
            gate_up,
            gate_up_experts,
            down,
            down_experts,
            intermediate,
            limit,
            fan: self.plane_fan(self.tensor(*h).rows),
            packed: self.tensor(*packed),
            h: self.tensor(*h),
            routed: self.tensor(*routed),
        };
        Some((p, *y))
    }

    /// The shared expert's gated add `gate_add` lands after the routed sum
    /// into `y`, with what was held back for it.
    fn shared_gate(
        &self,
        gate_add: Option<&poem_ir::Linear>,
        y: ValueId,
        banks: Option<HeldShared>,
        gate_w: Option<HeldGate>,
    ) -> Option<kernels_cuda::linear::moe::SharedGate> {
        match gate_add {
            Some(&poem_ir::Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y: out,
            }) if routed == y => Some(kernels_cuda::linear::moe::SharedGate {
                shared: self.tensor(shared),
                gate: self.tensor(gate),
                y: self.tensor(out),
                banks: banks.filter(|b| b.y == shared).map(|b| {
                    kernels_cuda::linear::moe::SharedBanks {
                        gate_up: self.tensor(b.gate_up),
                        down: self.tensor(b.down),
                    }
                }),
                gate_w: gate_w.filter(|g| g.gate == gate).map(|g| self.tensor(g.w)),
            }),
            _ => None,
        }
    }

    /// The routed MLP that is `v`'s one reader, in `v`'s region and reading
    /// `act`, with what it sums into and its gated add.
    fn routed_reader(
        &mut self,
        v: ValueId,
        act: ValueId,
    ) -> Option<(
        kernels_cuda::linear::moe::RoutedMlp,
        ValueId,
        poem_ir::Linear,
    )> {
        use poem_ir::{Linear, Operands as _, Operation};
        let poem_ir::Def::Op(at) = self.values()[v.0 as usize].def else {
            return None;
        };
        let nodes = self.nodes();
        let mut inputs = Vec::new();
        let mut readers = nodes
            .iter()
            .enumerate()
            .skip(at as usize + 1)
            .filter(|(_, node)| {
                inputs.clear();
                node.op.inputs(&mut inputs);
                inputs.contains(&v)
            });
        let (Some((reader_at, reader)), None) = (readers.next(), readers.next()) else {
            return None;
        };
        if !self.one_region(at, reader_at as u32) {
            return None;
        }
        let Operation::Fused(Fused::MoeExperts {
            select,
            glu,
            down,
            norm: None,
            sum,
            gate_add: Some(gate_add),
        }) = &reader.op
        else {
            return None;
        };
        if !matches!(**select, Linear::MoeMatmulSelect { x, .. } if x == act) {
            return None;
        }
        let (p, y) = self.routed_mlp(select, glu, down, sum)?;
        Some((p, y, (**gate_add).clone()))
    }

    /// Whether the shared expert MLP `act` → `held.y` is held back for the
    /// routed MLP that folds it in: that MLP is its one reader, reads `act`,
    /// and lands the shared expert as items of its own legs
    /// (`linear::moe::absorbs_shared`).
    pub(super) fn holds_shared(&mut self, act: ValueId, held: HeldShared) -> bool {
        if !self.dense_weight(held.gate_up) || !self.dense_weight(held.down) {
            return false;
        }
        let Some((p, y, gate_add)) = self.routed_reader(held.y, act) else {
            return false;
        };
        self.shared_gate(Some(&gate_add), y, Some(held), None)
            .filter(|s| s.banks.is_some())
            .is_some_and(|s| {
                kernels_cuda::linear::moe::absorbs_shared(self.ctx(), &p, s, self.tensor(y).width)
            })
    }

    /// Whether the shared expert's gate `act` · `held.w` is held back for
    /// the routed MLP's fold, as [`Run::holds_shared`].
    pub(super) fn holds_gate(&mut self, act: ValueId, held: HeldGate) -> bool {
        if !self.dense_weight(held.w) || self.tensor(held.w).rows != 1 {
            return false;
        }
        let Some((p, y, gate_add)) = self.routed_reader(held.gate, act) else {
            return false;
        };
        self.shared_gate(Some(&gate_add), y, None, Some(held))
            .filter(|s| s.gate_w.is_some())
            .is_some_and(|s| {
                kernels_cuda::linear::moe::absorbs_shared(self.ctx(), &p, s, self.tensor(y).width)
            })
    }

    /// A routed swiglu MLP over dense bf16 banks as one
    /// `linear::moe::decode_mlp` (which may fold `gate_add` in too, and land
    /// a shared expert held back for it) or `linear::moe::mlp_swiglu`, which
    /// keeps the routes in expert order across both matmuls. `None` where it
    /// left the legs to the caller, else whether it folded the gate add.
    fn moe_experts_ordered(
        &mut self,
        select: &poem_ir::Linear,
        glu: &poem_ir::Linear,
        down: &poem_ir::Linear,
        sum: &poem_ir::Linear,
        gate_add: Option<&poem_ir::Linear>,
    ) -> Result<Option<bool>, kernels_cuda::Error> {
        let held = self.shared_mlp.take();
        let held_gate = self.shared_gate_w.take();
        let routed = self.routed_mlp(select, glu, down, sum);
        let shared = routed.and_then(|(_, y)| self.shared_gate(gate_add, y, held, held_gate));
        let lands = shared.is_some_and(|s| {
            s.banks.is_some() == held.is_some() && s.gate_w.is_some() == held_gate.is_some()
        });
        let lost = || kernels_cuda::Error::Backend {
            op: "linear.moe_decode_mlp",
            detail: "a shared expert held back for a routed MLP that does not land it".into(),
        };
        let holds = held.is_some() || held_gate.is_some();
        if holds && !lands {
            return Err(lost());
        }
        let Some((p, y)) = routed else {
            return Ok(None);
        };
        if kernels_cuda::linear::moe::decode_mlp(self.ctx(), &p, shared, &mut self.tensor(y))? {
            return Ok(Some(shared.is_some()));
        }
        if holds {
            return Err(lost());
        }
        kernels_cuda::linear::moe::mlp_swiglu(self.ctx(), &p, &mut self.tensor(y))?;
        Ok(Some(false))
    }

    /// A fused op as the primitives it replaced, each over the values the
    /// fused op keeps.
    fn primitives(&mut self, op: &Fused) -> Result<(), KernelError> {
        use poem_exec::{DispatchAttention, DispatchElementwise, DispatchLayout, DispatchLinear};
        use poem_ir::{Attention, Elementwise, Layout, Linear};
        match op {
            Fused::MatmulSwiglu {
                act,
                w,
                intermediate,
                packed,
                y,
            } => {
                DispatchLinear::dispatch(
                    self,
                    &Linear::Matmul {
                        act: *act,
                        w: *w,
                        y: *packed,
                    },
                )?;
                DispatchLinear::dispatch(
                    self,
                    &Linear::MlpSwiglu {
                        packed: *packed,
                        intermediate: *intermediate,
                        y: *y,
                    },
                )
            }
            Fused::MatmulSplitRows {
                act,
                w,
                width,
                y,
                left,
                right,
            } => {
                DispatchLinear::dispatch(
                    self,
                    &Linear::Matmul {
                        act: *act,
                        w: *w,
                        y: *y,
                    },
                )?;
                DispatchLayout::dispatch(
                    self,
                    &Layout::SplitRows {
                        x: *y,
                        width: *width,
                        left: *left,
                        right: *right,
                    },
                )
            }
            Fused::MoeSelectGlu { select, glu } => {
                DispatchLinear::dispatch(self, select)?;
                DispatchLinear::dispatch(self, glu)
            }
            Fused::MoeExperts {
                select,
                glu,
                down,
                norm,
                sum,
                gate_add,
            } => {
                let ran = if norm.is_some() {
                    None
                } else {
                    self.moe_experts_ordered(select, glu, down, sum, gate_add.as_deref())
                        .map_err(crate::error::kernel)?
                };
                if ran.is_none() {
                    for leg in [select, glu, down] {
                        DispatchLinear::dispatch(self, leg)?;
                    }
                    if let Some(norm) = norm {
                        DispatchElementwise::dispatch(self, norm)?;
                    }
                    DispatchLinear::dispatch(self, sum)?;
                }
                match gate_add {
                    Some(gate_add) if ran != Some(true) => DispatchLinear::dispatch(self, gate_add),
                    _ => Ok(()),
                }
            }
            Fused::LseSink {
                attn,
                sink,
                head_dim,
                o_out,
            } => {
                let (Attention::DecodeLse { o, lse, .. } | Attention::PrefillLse { o, lse, .. }) =
                    **attn
                else {
                    return Err(crate::error::kernel(kernels_cuda::Error::Unsupported {
                        op: op.name(),
                    }));
                };
                DispatchAttention::dispatch(self, attn)?;
                DispatchAttention::dispatch(
                    self,
                    &Attention::Sink {
                        o,
                        lse,
                        sink: *sink,
                        head_dim: *head_dim,
                        o_out: *o_out,
                    },
                )
            }
            Fused::RopeYarnAppend {
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
                q_out,
                k_out,
            } => {
                DispatchElementwise::dispatch(
                    self,
                    &Elementwise::RopeYarn {
                        q: *q,
                        k: *k,
                        positions: *positions,
                        head_dim: *head_dim,
                        theta: *theta,
                        factor: *factor,
                        beta_fast: *beta_fast,
                        beta_slow: *beta_slow,
                        attention_factor: *attention_factor,
                        original_max_position: *original_max_position,
                        interleaved: *interleaved,
                        q_out: *q_out,
                        k_out: *k_out,
                    },
                )?;
                DispatchAttention::dispatch(
                    self,
                    &Attention::KvAppend {
                        k: *k_out,
                        v: *v,
                        cache: *cache,
                        write_page: *write_page,
                        write_offset: *write_offset,
                    },
                )
            }
            Fused::SplitRopeYarnAppend {
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
            } => {
                // The split lands q where the rotation leaves it.
                DispatchLayout::dispatch(
                    self,
                    &Layout::SplitQkv {
                        packed: *packed,
                        q_width: *q_width,
                        kv_width: *kv_width,
                        q: *q_out,
                        k: *k_out,
                        v: *v_out,
                    },
                )?;
                DispatchElementwise::dispatch(
                    self,
                    &Elementwise::RopeYarn {
                        q: *q_out,
                        k: *k_out,
                        positions: *positions,
                        head_dim: *head_dim,
                        theta: *theta,
                        factor: *factor,
                        beta_fast: *beta_fast,
                        beta_slow: *beta_slow,
                        attention_factor: *attention_factor,
                        original_max_position: *original_max_position,
                        interleaved: *interleaved,
                        q_out: *q_out,
                        k_out: *k_out,
                    },
                )?;
                DispatchAttention::dispatch(
                    self,
                    &Attention::KvAppend {
                        k: *k_out,
                        v: *v_out,
                        cache: *cache,
                        write_page: *write_page,
                        write_offset: *write_offset,
                    },
                )
            }
            _ => Err(crate::error::kernel(kernels_cuda::Error::Unsupported {
                op: op.name(),
            })),
        }
    }

    /// The gated-delta decode step whole where the kernel covers it, else
    /// its primitives over the values the fused op keeps.
    fn gdn_step(&mut self, op: &Fused) -> Result<(), KernelError> {
        use poem_exec::{DispatchAttention, DispatchLayout};
        let Fused::GdnStep {
            qkvz,
            width,
            qkv,
            z,
            conv_weight,
            conv_state,
            conv_width,
            dilation,
            conv,
            ba,
            dt_bias,
            a_log,
            gates,
            state,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            y,
        } = *op
        else {
            unreachable!("only the gated-delta step comes here");
        };
        if !self.rs_extended() {
            let done = kernels_cuda::ssm::gdn_step_fused(
                self.ctx(),
                self.tensor(qkvz),
                self.tensor(conv_weight),
                &self.recurrent_absolute(conv_state),
                conv_width,
                dilation,
                self.tensor(ba),
                self.tensor(dt_bias),
                self.tensor(a_log),
                &self.recurrent_absolute(state),
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                &mut self.tensor(qkv),
                &mut self.tensor(z),
                &mut self.tensor(y),
            )
            .map_err(crate::error::kernel)?;
            if done {
                self.rs_move("attention.ssm_causal_conv1d", qkv, self.tensor(qkv))
                    .map_err(crate::error::kernel)?;
                return self
                    .rs_move("attention.ssm_gdn_prep", ba, self.tensor(ba))
                    .map_err(crate::error::kernel);
            }
        }
        DispatchLayout::dispatch(
            self,
            &poem_ir::Layout::SplitRows {
                x: qkvz,
                width,
                left: qkv,
                right: z,
            },
        )?;
        DispatchAttention::dispatch(
            self,
            &poem_ir::Attention::SsmCausalConv1d {
                x: qkv,
                weight: conv_weight,
                bias: None,
                state: conv_state,
                conv_width,
                dilation,
                y: conv,
            },
        )?;
        DispatchAttention::dispatch(
            self,
            &poem_ir::Attention::SsmGdnPrep {
                ba,
                dt_bias,
                a_log,
                gates,
            },
        )?;
        DispatchAttention::dispatch(
            self,
            &poem_ir::Attention::SsmGatedDelta {
                qkv: conv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            },
        )
    }

    /// Qwen's gated attention prologue in one launch where the kernel takes
    /// it, else its primitives over the values the fused op keeps.
    fn q_gate_append(&mut self, op: &Fused) -> Result<(), KernelError> {
        use poem_exec::{DispatchAttention, DispatchElementwise, DispatchLayout};
        let Fused::QGateNormRopeAppend {
            qg,
            head_dim,
            q,
            gate,
            q_norm,
            q_eps,
            qn,
            k,
            k_norm,
            k_eps,
            kn,
            positions,
            rotary_dim,
            theta,
            span,
            q_out,
            k_out,
            v,
            cache,
            write_page,
            write_offset,
        } = *op
        else {
            unreachable!("only the gated attention prologue comes here");
        };
        if qkv_append::q_gate_k_norm_rope_append(
            self.ctx(),
            self.tensor(qg),
            self.tensor(k),
            self.tensor(v),
            self.tensor(q_norm),
            self.tensor(k_norm),
            self.tensor(positions),
            q_eps,
            k_eps,
            &self.pool(cache),
            self.tensor(write_page),
            self.tensor(write_offset),
            head_dim,
            rotary_dim,
            span,
            theta,
            &mut self.tensor(q_out),
            &mut self.tensor(gate),
            &mut self.tensor(q),
        )
        .map_err(crate::error::kernel)?
        {
            return Ok(());
        }
        DispatchLayout::dispatch(
            self,
            &poem_ir::Layout::SplitQGate {
                packed: qg,
                head_dim,
                q,
                gate,
            },
        )?;
        DispatchElementwise::dispatch(
            self,
            &poem_ir::Elementwise::RmsnormPerHeadPlusOne {
                x: q,
                weight: q_norm,
                head_dim,
                eps: q_eps,
                y: qn,
            },
        )?;
        DispatchElementwise::dispatch(
            self,
            &poem_ir::Elementwise::RmsnormPerHeadPlusOne {
                x: k,
                weight: k_norm,
                head_dim,
                eps: k_eps,
                y: kn,
            },
        )?;
        DispatchElementwise::dispatch(
            self,
            &poem_ir::Elementwise::RopePartial {
                q: qn,
                k: kn,
                positions,
                rotary_dim,
                head_dim,
                theta,
                span,
                q_out,
                k_out,
                freqs: None,
            },
        )?;
        DispatchAttention::dispatch(
            self,
            &poem_ir::Attention::KvAppend {
                k: k_out,
                v,
                cache,
                write_page,
                write_offset,
            },
        )
    }

    fn custom_cuda(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
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
                &mut self.tensor(*q),
            ),
            _ => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
        }
    }
}
