//! Kernel comptime params: the `K:V` list each kernel source is compiled with, per role kind,
//! from the plan, the floor and the build's levers. Replaces the shape half of `mkfull.py`,
//! `gemma/mkfull_gemma.py`, `g12/mkfull_g12.py`, `gptoss/mkfull_gptoss.py` and the prep's
//! `params.txt` writer in `prep_q35.py` / `prep.rs`.
//!
//! A kind's params ([`KernelBuild::params`]) hold for every PE of the kind. Where one PE of a kind
//! differs from the others (Gemma's shared-KV units take `SH 1` and their own `Ic`, Gemma 12B's
//! attention and plain rows their own `Hc` and `ATT`), [`tile`] resolves the PE's own list: the
//! kind's params, then what its role and the plan imply ([`role_params`]), then the family's
//! notes: keyed `U:<NAME>` anywhere in a unit, for every PE of the unit ([`UNIT_PREFIX`]: Gemma's
//! shared-KV units' `Ic`), and keyed `P:<NAME>` for the one PE ([`NOTE_PREFIX`]).

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::spatial::split::floor::Coord;
use crate::spatial::split::plan::Plan;
use crate::spatial::split::program::Program;
use crate::spatial::split::role::{PeRole, RoleKind, WeightPart};

/// One param value as the kernel source takes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Value {
    Int(i64),
    Bool(bool),
    /// An f32 literal as the layout writes it (Python's `repr`: gpt-oss's `QSC`).
    Float(String),
}

impl Value {
    /// As the layout and `link.sh` write it.
    pub fn render(&self) -> String {
        match self {
            Value::Int(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Float(s) => s.clone(),
        }
    }
}

/// An ordered `K:V` list (order kept: `link.sh` / cslc command lines and `params.txt` are
/// compared byte for byte).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Params {
    pub items: Vec<(String, Value)>,
}

impl Params {
    /// Sets `k` (in place when present, else appended).
    pub fn set(&mut self, k: &str, v: Value) {
        match self.items.iter_mut().find(|(n, _)| n == k) {
            Some((_, old)) => *old = v,
            None => self.items.push((k.to_string(), v)),
        }
    }

    pub fn get(&self, k: &str) -> Option<&Value> {
        self.items.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }

    pub fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }

    /// `K:V,K:V` as `link.sh` passes it.
    pub fn render(&self) -> String {
        self.items
            .iter()
            .map(|(k, v)| format!("{k}:{}", v.render()))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// A role kind's kernel: its source file under `csl/split`, the symbols it owns (the rest get
/// one-word dummies), its params.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KernelBuild {
    pub source: &'static str,
    pub owns: Vec<&'static str>,
    pub params: Params,
}

impl KernelBuild {
    /// `source`'s kernel owning `owns`, its params `items` in order.
    pub fn new(source: &'static str, owns: &[&'static str], items: Vec<(&str, Value)>) -> Self {
        let mut params = Params::default();
        for (k, v) in items {
            params.set(k, v);
        }
        KernelBuild {
            source,
            owns: owns.to_vec(),
            params,
        }
    }

    /// [`KernelBuild::new`] with integer params.
    pub fn ints(source: &'static str, owns: &[&'static str], items: &[(&str, u32)]) -> Self {
        let items = items
            .iter()
            .map(|&(k, v)| (k, Value::Int(i64::from(v))))
            .collect();
        KernelBuild::new(source, owns, items)
    }

    /// The filler: an empty kernel the layout writes inline (no source file, owns nothing).
    pub fn filler() -> Self {
        KernelBuild::new("", &[], Vec::new())
    }
}

/// The build's lever overrides on top of a family's defaults (`CYV`, `GELUT`, `WSPEC`, `SLW`,
/// `RING`, diagnostics ...): the cslc command line's extra `K:V`. A key `<kind>.K` (the kind's
/// [`RoleKind::name`]: `w.FIFOW:16`) sets the param of that kind only.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Levers {
    pub items: BTreeMap<String, i64>,
}

impl Levers {
    /// `K:V,K:V` (or `K=V`), as `link.sh`'s extra params and `split-geo --levers` take them.
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

/// The key prefix of a family's per-PE kernel param note (`P:SH`, `P:Ic`, ...): the note's value
/// is that PE's param, over its kind's.
pub const NOTE_PREFIX: &str = "P:";

/// The key prefix of a family's per-unit kernel param note (`U:Ic`): the value is the param of
/// every PE of the unit the note's PE is in, over its kind's (a PE's own `P:` note wins).
pub const UNIT_PREFIX: &str = "U:";

/// A program's kernel param notes: by PE and by unit index.
#[derive(Default)]
pub struct Notes {
    pub pe: HashMap<Coord, Vec<(&'static str, u32)>>,
    pub unit: HashMap<u32, Vec<(&'static str, u32)>>,
}

/// The kernel builds of every role kind a program uses (each family provides its
/// [`super::super::family::Family::kernels`]; this applies the levers and fills the shared shape
/// params). A lever sets the param of that name in every kind that has it; one no kind has is an
/// error (a misspelt lever would otherwise build the default silently).
pub fn kernels(prog: &Program, levers: &Levers) -> Result<BTreeMap<RoleKind, KernelBuild>, String> {
    let mut ks = prog.family.kernels(prog)?;
    for (key, &v) in &levers.items {
        let (only, k) = match key.split_once('.') {
            Some((kind, k)) => (Some(kind), k),
            None => (None, key.as_str()),
        };
        let mut hit = false;
        for (kind, kb) in ks.iter_mut() {
            if only.is_some_and(|o| o != kind.name()) {
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

/// The kernel param notes of a program ([`NOTE_PREFIX`], [`UNIT_PREFIX`]); a unit note outside
/// every unit is an error.
pub fn notes(prog: &Program) -> Result<Notes, String> {
    let mut m = Notes::default();
    for n in &prog.wiring.notes {
        if let Some(p) = n.key.strip_prefix(NOTE_PREFIX) {
            m.pe.entry(n.at).or_default().push((p, n.value));
        } else if let Some(p) = n.key.strip_prefix(UNIT_PREFIX) {
            let u = prog
                .floor
                .unit_at(n.at)
                .ok_or_else(|| format!("note {} at {:?}: in no unit", n.key, n.at))?;
            m.unit.entry(u.index).or_default().push((p, n.value));
        }
    }
    Ok(m)
}

/// What a PE's role and the plan imply for its kernel's params, where the kind's list does not
/// hold for every PE: a row run's dims (`Hc`) and role (`ATT`: 1 attention rows, 2 plain rows) on
/// the weight and CX PEs of plans with several runs; `SH 1` on the units whose layers all read
/// another layer's KV; a unit's MLP width a column (`Ic`) where the layers' widths differ; `NTW 1`
/// on the store PEs outside every storage block (Gemma's route-only pass PEs; a block's own
/// CX-column PEs compile its full NTW). Only params the kind takes are set.
pub fn role_params(prog: &Program, at: Coord, kind: &Params, out: &mut Params) {
    let plan = &prog.plan;
    let role = prog.floor.roles.get(at);
    let runs = &plan.partition.row_dims;
    let run = |r: u32| runs.iter().find(|x| x.rows.0 <= r && r < x.rows.1);
    let set_run = |r: u32, out: &mut Params| {
        if runs.len() > 1
            && kind.has("Hc")
            && let Some(x) = run(r)
        {
            out.set("Hc", Value::Int(i64::from(x.dims)));
        }
    };
    match role {
        PeRole::Weight(w) => {
            set_run(w.row, out);
            let att = match w.part {
                WeightPart::AttnRow => Some(1),
                WeightPart::PlainRow => Some(2),
                _ => None,
            };
            if let (Some(a), true) = (att, kind.has("ATT")) {
                out.set("ATT", Value::Int(a));
            }
            unit_params(plan, w.unit, kind, out);
        }
        PeRole::Cx(c) => set_run(c.row, out),
        PeRole::Cy(c) => unit_params(plan, c.unit, kind, out),
        PeRole::Store(st)
            if st.tokens == 0
                && kind.has("NTW")
                && !prog.floor.stores.iter().any(|b| b.rect.contains(at)) =>
        {
            out.set("NTW", Value::Int(1));
        }
        _ => {}
    }
}

/// A unit's own params: `SH 1` where its layers read another layer's KV, its MLP width a
/// column (`Ic`) where the plan states per-layer widths.
fn unit_params(plan: &Plan, u: u32, kind: &Params, out: &mut Params) {
    if kind.has("SH") && shared_unit(plan, u) {
        out.set("SH", Value::Int(1));
    }
    let mlp = &plan.shape.mlp;
    if kind.has("Ic") && !mlp.layer_widths.is_empty() {
        let w = mlp.width(u * plan.partition.layers_per_unit);
        out.set("Ic", Value::Int(i64::from(w / plan.partition.cols)));
    }
}

/// Whether every layer of unit `u` reads another layer's KV (Gemma's shared-KV units).
fn shared_unit(plan: &Plan, u: u32) -> bool {
    let k = plan.partition.layers_per_unit;
    let ls = &plan.shape.layers;
    let (a, b) = ((u * k) as usize, ((u + 1) * k) as usize);
    b <= ls.len()
        && ls[a..b]
            .iter()
            .all(|l| matches!(l, crate::spatial::split::plan::LayerKind::SharedKv { .. }))
}

/// The kernel kind and resolved params of the PE at `at` (`None`: no kernel build for its kind,
/// the empty filler).
pub fn tile(
    prog: &Program,
    kinds: &BTreeMap<RoleKind, KernelBuild>,
    notes: &Notes,
    at: Coord,
) -> Option<(RoleKind, Params)> {
    let role = prog.floor.roles.get(at);
    let kind = role.kind();
    let kb = kinds.get(&kind)?;
    let mut p = kb.params.clone();
    role_params(prog, at, &kb.params, &mut p);
    if !notes.unit.is_empty()
        && let Some(ns) = prog
            .floor
            .unit_at(at)
            .and_then(|u| notes.unit.get(&u.index))
    {
        for &(k, v) in ns {
            p.set(k, Value::Int(i64::from(v)));
        }
    }
    if let Some(ns) = notes.pe.get(&at) {
        for &(k, v) in ns {
            p.set(k, Value::Int(i64::from(v)));
        }
    }
    Some((kind, p))
}
