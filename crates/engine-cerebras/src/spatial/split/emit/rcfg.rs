//! rcfg: the per-PE words a kernel reads at init (its route table, then its queue colours and
//! role words, in the kernel's own ABI order). The route-table part is generic
//! ([`super::super::fabric::Fabric::route_words`]); the rest is each kernel's ABI, so each family
//! implements [`RcfgAbi`] for its kernels.

use crate::spatial::split::floor::{Coord, Grid};
use crate::spatial::split::program::Program;
use crate::spatial::split::role::RoleKind;

/// A family's rcfg ABI: the words each of its kernels reads.
pub trait RcfgAbi {
    /// Route slots the kernel of `kind` reserves (weight 16, Gemma's weight 18, relay 4, others
    /// 12).
    fn route_slots(&self, kind: RoleKind) -> usize;

    /// The full rcfg words of the PE at `at` (route table first). Errs when the PE's bindings are
    /// incomplete for its role.
    fn rcfg(&self, prog: &Program, at: Coord) -> Result<Vec<u32>, String>;
}

/// Every PE's rcfg words (`None` for PEs whose kernel reads none: idle).
pub fn all(prog: &Program) -> Result<Grid<Option<Vec<u32>>>, String> {
    let roles = &prog.floor.roles;
    let mut out = Grid::filled(roles.w, roles.h, None);
    for (at, role) in roles.iter() {
        if role.kind() == RoleKind::Idle {
            continue;
        }
        *out.get_mut(at) = Some(prog.family.rcfg(prog, at)?);
    }
    Ok(out)
}

/// The rcfg dump text two builds compare: one line a non-idle PE, `x y kind words...` with the kind's
/// [`RoleKind::name`], x outer, then y.
pub fn dump(prog: &Program) -> Result<String, String> {
    let words = all(prog)?;
    let roles = &prog.floor.roles;
    let mut out = String::new();
    for x in 0..roles.w {
        for y in 0..roles.h {
            let at = Coord::new(x, y);
            let Some(w) = words.get(at) else {
                continue;
            };
            out.push_str(&format!("{x} {y} {}", roles.get(at).kind().name()));
            for v in w {
                out.push_str(&format!(" {v}"));
            }
            out.push('\n');
        }
    }
    Ok(out)
}
