use crate::run::Run;

impl poem_exec::DispatchProbe for Run<'_> {
    fn enter(&mut self, node: &poem_ir::Node) {
        Run::enter(self, node);
    }

    fn probe(&mut self, node: &poem_ir::Node) {
        use poem_ir::Operands;
        self.walked(node);
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
        dump_rows(self, node);
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

fn dump_rows(run: &mut Run<'_>, node: &poem_ir::Node) {
    use poem_ir::Operands;
    use std::sync::atomic::{AtomicU32, Ordering};
    static STATE: AtomicU32 = AtomicU32::new(0);
    static COUNT: AtomicU32 = AtomicU32::new(0);
    static LAST: AtomicU32 = AtomicU32::new(0);
    if STATE.load(Ordering::Relaxed) >= 2 {
        return;
    }
    let stream = run.ctx().stream();
    if crate::device::alloc::is_capturing(stream) {
        return;
    }
    let layer = node.layer.unwrap_or(0);
    if STATE.load(Ordering::Relaxed) == 1 && layer < LAST.load(Ordering::Relaxed) {
        STATE.store(2, Ordering::Relaxed);
        return;
    }
    let mut outs = Vec::new();
    node.op.outputs(&mut outs);
    for id in outs {
        let Some(decl) = run.values().get(id.0 as usize) else {
            continue;
        };
        let poem_ir::Ty::Tensor { dtype, .. } = &decl.ty else {
            continue;
        };
        let elem: usize = match dtype {
            poem_ir::Dtype::Bf16 => 2,
            poem_ir::Dtype::F32 => 4,
            _ => continue,
        };
        if !matches!(decl.def, poem_ir::Def::Op(_) | poem_ir::Def::Merge(_)) || !run.resolvable(id) {
            continue;
        }
        let t = run.tensor(id);
        if t.ptr == 0 || t.rows == 0 || t.width == 0 {
            continue;
        }
        if STATE.load(Ordering::Relaxed) == 0 {
            if t.rows < 32 {
                continue;
            }
            STATE.store(1, Ordering::Relaxed);
        }
        LAST.store(layer, Ordering::Relaxed);
        let rows = (t.rows as usize).min(3);
        let bytes = rows * t.width as usize * elem;
        let mut host = vec![0u8; bytes];
        let _ = crate::device::ctx::sync(stream);
        if crate::device::copy_any(stream, host.as_mut_ptr() as u64, t.ptr, bytes).is_err() {
            continue;
        }
        let _ = crate::device::ctx::sync(stream);
        let vals: Vec<f32> = match dtype {
            poem_ir::Dtype::F32 => host.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            _ => host
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                .collect(),
        };
        let mut line = format!(
            "dump: {} {} layer={:?} v={} {}x{}",
            COUNT.fetch_add(1, Ordering::Relaxed),
            node.op.name(),
            node.layer,
            id.0,
            t.rows,
            t.width
        );
        for r in vals.chunks(t.width as usize) {
            let sum: f64 = r.iter().map(|v| f64::from(*v)).sum();
            let abs: f64 = r.iter().map(|v| f64::from(v.abs())).sum();
            line.push_str(&format!(
                " | {sum:.5e} {abs:.5e} {:.4} {:.4} {:.4}",
                r[0],
                r.get(1).copied().unwrap_or(0.0),
                r.get(r.len() / 2).copied().unwrap_or(0.0)
            ));
        }
        eprintln!("{line}");
        if COUNT.load(Ordering::Relaxed) > 3000 {
            STATE.store(2, Ordering::Relaxed);
        }
    }
}
