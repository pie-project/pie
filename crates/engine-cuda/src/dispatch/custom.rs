use kernels_cuda::custom;
use poem_exec::{DispatchFused, KernelError};
use poem_ir::{Fused, Operands};

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        match op {
            Fused::QkvFusedQknormRopeVnormWrite { .. } => self.custom_cuda(op),
            Fused::RopeYarnAppend { .. } => self.rope_yarn_append(op),
            Fused::LseSink { .. } => self.lse_sink(op),
            Fused::MatmulGeglu { .. }
            | Fused::MatmulSwiglu { .. }
            | Fused::MlpSwiglu { .. }
            | Fused::LmHeadSoftcap { .. }
            | Fused::MatmulBias { .. }
            | Fused::MoeSelectGlu { .. }
            | Fused::MoeExperts { .. } => self.fused_linear(op),
            Fused::ResidualAddRmsnorm { .. }
            | Fused::RmsnormResidualAdd { .. }
            | Fused::EmbedScaleAdd { .. }
            | Fused::EmbedScaleAddSelect { .. }
            | Fused::RmsnormRopePartialQ { .. }
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
