use kernels_cuda::custom;
use model_exec::{DispatchCustomCuda, KernelError};
use model_ir::CustomCuda;

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        self.custom_cuda(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn custom_cuda(&mut self, op: &CustomCuda) -> Result<(), kernels_cuda::Error> {
        match op {
            CustomCuda::QkvFusedQknormRopeVnormWrite {
                packed,
                positions,
                q_norm_weight,
                q_norm_eps,
                k_norm_weight,
                k_norm_eps,
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
                *q_norm_eps,
                self.tensor(*k_norm_weight),
                *k_norm_eps,
                &self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
                *kv_heads,
                *head_dim,
                *theta,
                *rotary_dim,
                &mut self.tensor(*q),
            ),
        }
    }
}

impl model_exec::DispatchProbe for Run<'_> {
    fn probe(&mut self, node: &model_ir::Node) {
        use model_ir::Operands;
        let tag = crate::record::PTR_TAG.load(std::sync::atomic::Ordering::Relaxed);
        if tag != 0 {
            let mut ins = Vec::new();
            let mut outs = Vec::new();
            node.op.inputs(&mut ins);
            node.op.outputs(&mut outs);
            let show = |run: &Self, ids: &[model_ir::ValueId]| -> String {
                ids.iter()
                    .map(|id| {
                        let Some(decl) = run.values().get(id.0 as usize) else {
                            return format!("{}:?", id.0);
                        };
                        match &decl.def {
                            model_ir::Def::Op(_)
                            | model_ir::Def::Merge(_)
                            | model_ir::Def::Weight(_)
                                if run.resolvable(*id) =>
                            {
                                let t = run.tensor(*id);
                                format!("{}:{:#x}/{}x{}", id.0, t.ptr, t.rows, t.width)
                            }
                            model_ir::Def::Op(_) | model_ir::Def::Merge(_) => {
                                format!("{}:unslotted", id.0)
                            }
                            model_ir::Def::Weight(_) => format!("{}:planes", id.0),
                            model_ir::Def::Input(kind) => {
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
            let model_ir::Ty::Tensor { dtype, .. } = &decl.ty else {
                continue;
            };
            let elem: usize = match dtype {
                model_ir::Dtype::Bf16 | model_ir::Dtype::F16 => 2,
                model_ir::Dtype::F32 => 4,
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
                model_ir::Dtype::F32 => host
                    .chunks_exact(4)
                    .position(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite()),
                model_ir::Dtype::Bf16 => host
                    .chunks_exact(2)
                    .position(|c| (u16::from_le_bytes([c[0], c[1]]) & 0x7f80) == 0x7f80),
                _ => host
                    .chunks_exact(2)
                    .position(|c| (u16::from_le_bytes([c[0], c[1]]) & 0x7c00) == 0x7c00),
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
