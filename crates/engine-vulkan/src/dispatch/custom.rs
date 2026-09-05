use model_exec::{DispatchCustomCuda, KernelError};
use model_ir::{CustomCuda, Operands};

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl model_exec::DispatchProbe for Run<'_> {
    fn probe(&mut self, node: &model_ir::Node) {
        use std::io::Write;
        let Some(dir) = crate::probe::dir() else {
            return;
        };
        if let Err(fault) = crate::probe::flush() {
            eprintln!("probe: flush failed: {fault}");
            return;
        }
        let mut outs = Vec::new();
        node.op.outputs(&mut outs);
        let seq = crate::probe::next_seq();
        let mut manifest = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(dir.join("manifest.txt"))
            .ok();
        for id in outs {
            let Some(decl) = self.values().get(id.0 as usize) else {
                continue;
            };
            let model_ir::Ty::Tensor { dtype, .. } = &decl.ty else {
                continue;
            };
            let elem: u64 = match dtype {
                model_ir::Dtype::Bf16 | model_ir::Dtype::F16 => 2,
                model_ir::Dtype::F32 | model_ir::Dtype::I32 | model_ir::Dtype::U32 => 4,
                _ => continue,
            };
            let t = self.tensor(id);
            let total = u64::from(t.rows) * u64::from(t.width) * elem;
            if total == 0 || t.buf == kernels_vulkan::ABSENT {
                continue;
            }
            let Ok(bytes) = self.handles().read(t.buf, total) else {
                continue;
            };
            let name = format!(
                "{seq:04}_{}_L{}_v{}.bin",
                node.op.name().replace(['.', '/'], "_"),
                node.layer.map_or(-1, i64::from),
                id.0
            );
            let _ = std::fs::write(dir.join(&name), &bytes);
            if let Some(m) = manifest.as_mut() {
                let _ = writeln!(
                    m,
                    "{seq} {} {} v{} {:?} {} {} {name}",
                    node.op.name(),
                    node.layer.map_or(-1, i64::from),
                    id.0,
                    dtype,
                    t.rows,
                    t.width
                );
            }
        }
    }
}
