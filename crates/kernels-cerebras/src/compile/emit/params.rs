//! Kernel comptime params: each kind's `K:V` list (its [`KernelBuild`], the build's levers on
//! top), and each PE's own list (its kind's, then the blocks' [`super::super::abi::Overrides`]
//! at that PE).

use std::collections::BTreeMap;

use serde::Serialize;

use crate::compile::abi::{Kernels, Params, Value};
use crate::compile::floor::Coord;
use crate::compile::program::Program;
use crate::compile::role::Kind;

/// The build's lever overrides on top of the kernels' defaults (`CYV`, `GELUT`, `WSPEC`, `SLW`,
/// `RING`, diagnostics ...): the cslc command line's extra `K:V`. A key `<kind>.K` (the kind's
/// [`Kind::name`]: `w.FIFOW:16`) sets the param of that kind only.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Levers {
    pub items: BTreeMap<String, i64>,
}

impl Levers {
    /// A plan's levers with `more` set over them.
    pub fn of(plan: &str, more: &str) -> Result<Levers, String> {
        let mut l = Levers::parse(plan)?;
        l.items.extend(Levers::parse(more)?.items);
        Ok(l)
    }

    /// `K:V,K:V` (or `K=V`), as `wafer-compile --levers` takes them.
    pub fn parse(s: &str) -> Result<Levers, String> {
        let mut items = BTreeMap::new();
        for kv in s.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv
                .split_once(':')
                .or_else(|| kv.split_once('='))
                .ok_or_else(|| format!("lever {kv}: K:V"))?;
            let n: i64 = v.parse().map_err(|e| format!("lever {kv}: {e}"))?;
            items.insert(k.to_string(), n);
        }
        Ok(Levers { items })
    }
}

/// The program's kernel builds with the levers applied. A lever sets the param of that name in
/// every kind that has it; one no kind has is an error (a misspelt lever would otherwise build
/// the default silently).
pub fn kernels(prog: &Program, levers: &Levers) -> Result<Kernels, String> {
    let mut ks = prog.kernels.clone();
    for (key, &v) in &levers.items {
        let (only, k) = match key.split_once('.') {
            Some((kind, k)) => (Some(kind), k),
            None => (None, key.as_str()),
        };
        let mut hit = false;
        for (kind, kb) in ks.iter_mut() {
            if only.is_some_and(|o| o != kind.name) {
                continue;
            }
            if let Some(old) = kb.params.get(k) {
                let nv = match old {
                    Value::Bool(_) => Value::Bool(v != 0),
                    _ => Value::Int(v),
                };
                kb.params.set(k, nv);
                hit = true;
            }
        }
        if !hit {
            return Err(format!("lever {key}: no kernel of this program takes it"));
        }
    }
    Ok(ks)
}

/// The kernel kind and resolved params of the PE at `at` (`None`: no kernel build for its kind,
/// the empty filler): its kind's params, then the overrides at `at` (each must be a param the
/// kind takes).
pub fn tile(prog: &Program, kinds: &Kernels, at: Coord) -> Result<Option<(Kind, Params)>, String> {
    let kind = prog.floor.roles.get(at).kind;
    let Some(kb) = kinds.get(&kind) else {
        return Ok(None);
    };
    let mut p = kb.params.clone();
    for (k, v) in prog.overrides.pe.get(&at).into_iter().flatten() {
        if !p.has(k.0) {
            return Err(format!(
                "{at:?}: override {} not a param of {}",
                k.0, kind.name
            ));
        }
        p.set(k.0, v.clone());
    }
    Ok(Some((kind, p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::target::{SHIPPING, searched};

    /// The numerics every shipping pick emits whatever its partition: the f16 SiLU without the
    /// subnormal flush (`SLW` 1; flushed, Qwen3-8B's logits sit ~5x further from f32 and its greedy
    /// run leaves the golden at a 0.017 margin).
    #[test]
    fn shipping_picks_emit_the_numerics_defaults() {
        let mut silu16 = 0;
        for name in SHIPPING {
            let plan = searched(name).plan;
            let levers = Levers::of(&plan.levers, "").unwrap();
            let prog = Program::of(plan).unwrap_or_else(|e| panic!("{name}: {e}"));
            let ks = kernels(&prog, &levers).unwrap();
            for (kind, kb) in &ks {
                if kb.params.get("SILU16") == Some(&Value::Int(1)) {
                    silu16 += 1;
                    assert_eq!(
                        kb.params.get("SLW"),
                        Some(&Value::Int(1)),
                        "{name}: {} runs the f16 SiLU",
                        kind.name
                    );
                }
            }
        }
        assert!(silu16 > 0, "no shipping pick runs the f16 SiLU");
    }
}
