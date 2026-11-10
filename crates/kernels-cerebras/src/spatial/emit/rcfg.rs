//! Every PE's rcfg words: its kind's kernel's writer ([`super::super::abi::Kernel::rcfg`]) over
//! the PE's bindings ([`super::super::kernels::Pe`]).

use crate::spatial::floor::{Coord, Grid};
use crate::spatial::kernels::Pe;
use crate::spatial::program::Program;

/// The rcfg words of the PE at `at` (its route table first).
pub fn of(prog: &Program, at: Coord) -> Result<Vec<u32>, String> {
    let kind = prog.floor.roles.get(at).kind;
    let f = prog
        .kernels
        .get(&kind)
        .and_then(|kb| kb.rcfg)
        .ok_or_else(|| format!("{at:?}: kind {} has no rcfg writer", kind.name))?;
    f(&Pe::new(prog, at))
}

/// Every PE's rcfg words (`None` for idle PEs, whose kernel reads none).
pub fn all(prog: &Program) -> Result<Grid<Option<Vec<u32>>>, String> {
    let roles = &prog.floor.roles;
    let mut out = Grid::filled(roles.w, roles.h, None);
    for (at, role) in roles.iter() {
        if role.is_idle() {
            continue;
        }
        *out.get_mut(at) = Some(of(prog, at)?);
    }
    Ok(out)
}

/// The rcfg dump text two builds compare: one line a non-idle PE, `x y kind words...` with the
/// kind's name, x outer, then y.
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
            out.push_str(&format!("{x} {y} {}", roles.get(at).kind.name));
            for v in w {
                out.push_str(&format!(" {v}"));
            }
            out.push('\n');
        }
    }
    Ok(out)
}
