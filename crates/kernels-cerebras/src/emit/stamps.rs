//! The phase stamps a diagnostics build leaves on the wafer (`TSTAMP:1`): which PEs the host
//! reads `tdbg` from after a launch, how many words, and how to read them. Written as
//! `stamps.json` beside the artifact only when some kernel stamps its phases, so a shipping build
//! emits nothing new; `wafer-bench` reads it back into per-layer and per-phase cycles.
//!
//! Each kernel source's `tdbg` length is its own (a fact of the source, as its params are):
//! - `weight.csl` (`TSTAMP` 1): `1 + 9 K` (MIXK 2 `1 + 6 K`, MIXK 3 `1 + 4 K`), from the pass start;
//! - `cy.csl` (`TSTAMP` 1): `1 + 6 K` (MIXK 2 / 3 `1 + 4 K`), from the pass start; MIXK 2's last `K`
//!   words are counts, not stamps (each layer's redo passes over the launch, always on);
//! - `cx.csl` (always on): `K + 2`, the pass start, each layer's end and the send (NRM 2: `5 K + 2`);
//! - `head_ctl.csl` (always on): 70, the pass start, the argmax, the root's winner of each step
//!   (`2 + s`, s < 64) and four phase ends (66 ..= 69);
//! - `head_pe.csl`, `head_pe_gptoss.csl` (always on): 6, a step's phase ends.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::abi::{Kernels, Params, Value};
use crate::emit::params::tile;
use crate::floor::Coord;
use crate::program::Program;

type R<T> = Result<T, String>;

/// The file beside the artifact.
pub const FILE: &str = "stamps.json";

/// How a probe's words read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Read {
    /// A CX column: the pass start, then `layers` layer ends and the send (each a difference).
    Layers { layers: u32 },
    /// Phase ends, each from word 0.
    Phases,
    /// The head control root: the steps' winners (`2 + s`) and its phase ends (0, 66 ..= 69).
    Loop,
}

/// One PE's `tdbg`, read after a launch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Probe {
    pub name: String,
    pub at: Coord,
    pub words: u32,
    /// The last words that count rather than stamp (read as they are).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub counts: u32,
    pub read: Read,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// Of a PE's `tdbg` words, the trailing ones that are counts (cy.csl MIXK 2: the redo passes).
pub fn counts(source: &str, p: &Params) -> u32 {
    match source {
        "cy.csl" if int(p, "MIXK") == Some(2) => int(p, "K").unwrap_or(0) as u32,
        _ => 0,
    }
}

fn int(p: &Params, k: &str) -> Option<i64> {
    match p.get(k)? {
        Value::Int(v) => Some(*v),
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Float(_) => None,
    }
}

/// The `tdbg` words a PE of `source` compiled with `p` stamps (`None`: none, or stamps off).
pub fn words(source: &str, p: &Params) -> Option<u32> {
    let k = int(p, "K").unwrap_or(0) as u32;
    let on = int(p, "TSTAMP") == Some(1);
    let mixk = int(p, "MIXK").unwrap_or(0);
    match source {
        "weight.csl" if on => Some(match mixk {
            2 => 1 + 6 * k,
            3 => 1 + 4 * k,
            _ => 1 + 9 * k,
        }),
        "cy.csl" if on => Some(if mixk <= 1 { 1 + 6 * k } else { 1 + 4 * k }),
        "cx.csl" => Some(if int(p, "NRM") == Some(2) {
            5 * k + 2
        } else {
            k + 2
        }),
        "head_ctl.csl" => Some(70),
        "head_pe.csl" | "head_pe_gptoss.csl" => Some(6),
        _ => None,
    }
}

/// Whether any kernel of the build stamps its phases (`TSTAMP` 1).
pub fn on(kinds: &Kernels) -> bool {
    kinds
        .values()
        .any(|kb| int(&kb.params, "TSTAMP") == Some(1))
}

/// The probes of a stamping build: every unit's CX column (its first weight row), the token
/// loop's root and, in unit 0 and the head, each stamping kind's first, middle and last PE
/// and the one nearest the CX column.
pub fn probes(prog: &Program, kinds: &Kernels) -> R<Vec<Probe>> {
    let f = &prog.floor;
    let mut out = Vec::new();
    let mut add = |name: String, at: Coord, read: Read| -> R<()> {
        if let Some((_, p)) = tile(prog, kinds, at)? {
            let kb = &kinds[&f.roles.get(at).kind];
            if let Some(words) = words(kb.source, &p)
                && !out.iter().any(|q: &Probe| q.at == at)
            {
                out.push(Probe {
                    name,
                    at,
                    words,
                    counts: counts(kb.source, &p),
                    read,
                });
            }
        }
        Ok(())
    };
    for u in &f.units {
        let at = Coord::new(u.cx_x(), u.row_y(0));
        let layers = u.layers.1 - u.layers.0;
        add(format!("cx.u{}", u.index), at, Read::Layers { layers })?;
    }
    add("head_ctl".into(), f.head.root, Read::Loop)?;
    let mut areas = Vec::new();
    if let Some(u0) = f.units.first() {
        areas.push((u0.rect, Some(u0.cx_x())));
    }
    areas.push((f.head.rect, None));
    for (r, cx) in areas {
        let mut by_kind: Vec<(&str, Vec<Coord>)> = Vec::new();
        for y in r.y..r.y + r.h {
            for x in r.x..r.x + r.w {
                let at = Coord::new(x, y);
                let kind = f.roles.get(at).kind;
                let Some(kb) = kinds.get(&kind) else { continue };
                if kb.source == "cx.csl" || kb.source == "head_ctl.csl" {
                    continue;
                }
                match by_kind.iter_mut().find(|(n, _)| *n == kind.name) {
                    Some((_, v)) => v.push(at),
                    None => by_kind.push((kind.name, vec![at])),
                }
            }
        }
        for (name, pes) in by_kind {
            let first = pes[0];
            let mut pick = vec![first, pes[pes.len() / 2], pes[pes.len() - 1]];
            if let Some(cx) = cx {
                let row: Vec<&Coord> = pes.iter().filter(|c| c.y == first.y).collect();
                if let Some(near) = row.iter().min_by_key(|c| c.x.abs_diff(cx)) {
                    pick.push(**near);
                }
            }
            for at in pick {
                add(format!("{name}.{}_{}", at.x, at.y), at, Read::Phases)?;
            }
        }
    }
    Ok(out)
}

/// The probes written beside a stamping build's artifact (`None`: not a stamping build).
pub fn read(artifact: &Path) -> R<Option<Vec<Probe>>> {
    let p = artifact.join(FILE);
    if !p.exists() {
        return Ok(None);
    }
    let s = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s)
        .map(Some)
        .map_err(|e| format!("{}: {e}", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::params::{Levers, kernels};
    use crate::target::{Ask, plan_shape};

    fn tiny(file: &str, preset: &str) -> Program {
        let path = format!(
            "{}/tools/wafer/baseline/shapes/{file}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let s = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        Program::of(plan_shape(s, &Ask::preset(preset)).unwrap().plan).unwrap()
    }

    /// A shipping build stamps nothing on the path; `TSTAMP:1` probes every unit's CX column,
    /// the loop's root and unit 0's weight and CY PEs, each at its source's `tdbg` length.
    #[test]
    fn a_stamping_build_probes_every_unit_and_its_phases() {
        for (file, preset) in [("tiny_llama", "tiny-q3"), ("tiny_gptoss", "tiny")] {
            let prog = tiny(file, preset);
            let plain = kernels(&prog, &Levers::of(&prog.plan.levers, "").unwrap()).unwrap();
            assert!(!on(&plain), "{file}");
            let ks = kernels(&prog, &Levers::of(&prog.plan.levers, "TSTAMP:1").unwrap()).unwrap();
            assert!(on(&ks), "{file}");
            let p = probes(&prog, &ks).unwrap();
            let cx: Vec<&Probe> = p.iter().filter(|q| q.name.starts_with("cx.u")).collect();
            assert_eq!(cx.len(), prog.floor.units.len(), "{file}");
            for (q, u) in cx.iter().zip(&prog.floor.units) {
                assert_eq!(
                    q.read,
                    Read::Layers {
                        layers: u.layers.1 - u.layers.0
                    }
                );
            }
            assert_eq!(
                p.iter().filter(|q| q.read == Read::Loop).count(),
                1,
                "{file}"
            );
            assert!(
                p.iter().any(|q| q.read == Read::Phases && q.words > 6),
                "{file}"
            );
        }
    }
}
