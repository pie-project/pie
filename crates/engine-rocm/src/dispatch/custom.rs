use kernels_rocm::custom;
use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};

use crate::run::Run;

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
    /// banks, a swiglu or its clamp, the weighted sum) as one
    /// `linear::moe::mlp_swiglu`, which keeps the routes in expert order
    /// across both matmuls where that measured faster. Whether it took the
    /// legs; the caller runs them otherwise.
    fn moe_experts_ordered(
        &mut self,
        select: &poem_ir::Linear,
        glu: &poem_ir::Linear,
        down: &poem_ir::Linear,
        sum: &poem_ir::Linear,
    ) -> Result<bool, kernels_rocm::Error> {
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
            return Ok(false);
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
            _ => return Ok(false),
        };
        if routes != down_routes || glu_in != *packed || glu_out != *h || summed != routed {
            return Ok(false);
        }
        let (gate_up, gate_up_experts) = self.expert_bank(*gate_up);
        let (down, down_experts) = self.expert_bank(*down_bank);
        let p = kernels_rocm::linear::moe::RoutedMlp {
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
        kernels_rocm::linear::moe::mlp_swiglu(self.ctx(), &p, &mut self.tensor(*y))?;
        Ok(true)
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
                sum,
                gate_add,
            } => {
                if !self
                    .moe_experts_ordered(select, glu, down, sum)
                    .map_err(crate::error::kernel)?
                {
                    for leg in [select, glu, down, sum] {
                        DispatchLinear::dispatch(self, leg)?;
                    }
                }
                match gate_add {
                    Some(gate_add) => DispatchLinear::dispatch(self, gate_add),
                    None => Ok(()),
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
                    return Err(crate::error::kernel(kernels_rocm::Error::Unsupported {
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
            _ => Err(crate::error::kernel(kernels_rocm::Error::Unsupported {
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
            let done = kernels_rocm::attn::ssm::gdn_step_fused(
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
        if custom::q_gate_k_norm_rope_append(
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

    fn custom_cuda(&mut self, op: &Fused) -> Result<(), kernels_rocm::Error> {
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
            _ => Err(kernels_rocm::Error::Unsupported { op: op.name() }),
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
