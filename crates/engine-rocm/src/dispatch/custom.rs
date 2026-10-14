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
