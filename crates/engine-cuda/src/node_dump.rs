//! A diagnostic: every node's outputs written to a directory while armed
//! (`set`), one file a value, in walk order. A probe for per-op diffs.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
static SEQ: AtomicUsize = AtomicUsize::new(0);

/// Arm (`Some`) or disarm the dump.
pub fn set(dir: Option<PathBuf>) {
    if let Some(dir) = &dir {
        let _ = std::fs::create_dir_all(dir);
    }
    SEQ.store(0, Ordering::Relaxed);
    *DIR.lock().expect("dump lock") = dir;
}

pub(crate) fn dump(run: &crate::run::Run<'_>, node: &poem_ir::Node) {
    use poem_ir::Operands;
    let Some(dir) = DIR.lock().expect("dump lock").clone() else {
        return;
    };
    let mut outs = Vec::new();
    node.op.outputs(&mut outs);
    for id in outs {
        let Some(decl) = run.values().get(id.0 as usize) else {
            continue;
        };
        let poem_ir::Ty::Tensor { dtype, .. } = &decl.ty else {
            continue;
        };
        let (elem, kind) = match dtype {
            poem_ir::Dtype::Bf16 => (2, "bf16"),
            poem_ir::Dtype::F32 => (4, "f32"),
            _ => continue,
        };
        if !matches!(decl.def, poem_ir::Def::Op(_) | poem_ir::Def::Merge(_)) || !run.resolvable(id)
        {
            continue;
        }
        let t = run.tensor(id);
        let total = (t.rows as usize * t.width as usize * elem).min(64 << 20);
        if total == 0 || t.ptr == 0 {
            continue;
        }
        let mut host = vec![0u8; total];
        if crate::device::copy_any(run.ctx().stream(), host.as_mut_ptr() as u64, t.ptr, total)
            .is_err()
        {
            continue;
        }
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            "{seq:05}_{}_{}_{}x{}.{kind}",
            node.op.name(),
            id.0,
            t.rows,
            t.width
        );
        let _ = std::fs::write(dir.join(name), host);
    }
}
