//! Kernel comptime params: each kind's `K:V` list (its [`KernelBuild`] from the family, the
//! build's levers on top), and each PE's own list (its kind's, then the blocks'
//! [`super::super::abi::Overrides`] at that PE).

use std::collections::BTreeMap;

use serde::Serialize;

use crate::spatial::abi::{Kernels, Params, Value};
use crate::spatial::floor::Coord;
use crate::spatial::program::Program;
use crate::spatial::role::Kind;

/// The build's lever overrides on top of a family's defaults (`CYV`, `GELUT`, `WSPEC`, `SLW`,
/// `RING`, diagnostics ...): the cslc command line's extra `K:V`. A key `<kind>.K` (the kind's
/// [`Kind::name`]: `w.FIFOW:16`) sets the param of that kind only.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Levers {
    pub items: BTreeMap<String, i64>,
}

impl Levers {
    /// `K:V,K:V` (or `K=V`), as `link.sh`'s extra params and `wafer-compile --levers` take them.
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
