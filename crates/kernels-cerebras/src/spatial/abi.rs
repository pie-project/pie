//! What a kernel is to the build: one [`Kernel`] descriptor a CSL source (its file, the symbols
//! it owns, its route table, its rcfg writer) and one typed params struct ([`params!`]: the
//! `@set_tile_code` fields in order). A program's [`Kernels`] map each [`Kind`] to a
//! [`KernelBuild`]: the descriptor with its params from the plan. Per-PE param differences are
//! typed [`Overrides`]; an rcfg word layout is a [`words!`] struct.
//!
//! The kernels themselves (descriptors, params, rcfg writers) are in [`super::kernels`].

use std::collections::BTreeMap;

use serde::Serialize;

use super::floor::Coord;
use super::kernels::Pe;
use super::role::Kind;

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

impl From<u32> for Value {
    fn from(v: u32) -> Value {
        Value::Int(i64::from(v))
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Value {
        Value::Bool(v)
    }
}

/// An f32 param as its literal ([`Value::Float`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lit(pub String);

impl From<Lit> for Value {
    fn from(v: Lit) -> Value {
        Value::Float(v.0)
    }
}

/// A param's name (a `@set_tile_code` field). Each params struct declares its own as
/// associated consts (`WeightQ35::ATT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Param(pub &'static str);

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

/// A kernel's rcfg writer: the PE's full rcfg words (its route table first).
pub type RcfgFn = fn(&Pe) -> Result<Vec<u32>, String>;

/// One kernel source under `programs/split`.
pub struct Kernel {
    /// The source file (empty: the filler, written inline).
    pub source: &'static str,
    /// The exported symbols it owns (the rest get one-word dummies).
    pub owns: &'static [&'static str],
    /// The params it compiles with, in `@set_tile_code` order (its params struct's `NAMES`).
    pub params: &'static [&'static str],
    /// Routes its rcfg's route table holds.
    pub route_slots: usize,
    /// Its rcfg words (`None`: it reads no rcfg).
    pub rcfg: Option<RcfgFn>,
}

/// The empty kernel of PEs with no role.
pub const FILLER: Kernel = Kernel {
    source: "",
    owns: &[],
    params: &[],
    route_slots: 0,
    rcfg: None,
};

/// A kind's kernel and its params.
#[derive(Clone, Debug, Serialize)]
pub struct KernelBuild {
    pub source: &'static str,
    pub owns: Vec<&'static str>,
    pub params: Params,
    #[serde(skip)]
    pub route_slots: usize,
    #[serde(skip)]
    pub rcfg: Option<RcfgFn>,
}

impl KernelBuild {
    /// `kernel` compiled with `params`; errs when they are not the kernel's params in its order.
    pub fn of(kernel: &Kernel, params: impl Into<Params>) -> Result<KernelBuild, String> {
        let params = params.into();
        let names: Vec<&str> = params.items.iter().map(|(n, _)| n.as_str()).collect();
        if names != kernel.params {
            return Err(format!(
                "{}: params {names:?}, the kernel declares {:?}",
                kernel.source, kernel.params
            ));
        }
        Ok(KernelBuild {
            source: kernel.source,
            owns: kernel.owns.to_vec(),
            params,
            route_slots: kernel.route_slots,
            rcfg: kernel.rcfg,
        })
    }
}

/// Every kind's kernel build of a program.
pub type Kernels = BTreeMap<Kind, KernelBuild>;

/// Per-PE param values over their kind's (a shared-KV unit's `SH`, a row run's `Hc`). Strict:
/// a param the PE's kernel does not take is an error at resolution.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Overrides {
    pub pe: BTreeMap<Coord, Vec<(Param, Value)>>,
}

impl Overrides {
    pub fn set(&mut self, at: Coord, p: Param, v: impl Into<Value>) {
        let e = self.pe.entry(at).or_default();
        let v = v.into();
        match e.iter_mut().find(|(q, _)| *q == p) {
            Some((_, old)) => *old = v,
            None => e.push((p, v)),
        }
    }
}

/// A kernel's params struct: one field a `@set_tile_code` param, in its order, typed (`u32`,
/// `bool`, [`Lit`]); its names as associated [`Param`] consts and `NAMES`; into [`Params`].
macro_rules! params {
    ($(#[$m:meta])* $v:vis struct $N:ident { $($f:ident : $t:ty),* $(,)? }) => {
        $(#[$m])*
        #[allow(non_snake_case)]
        #[derive(Clone, Debug)]
        $v struct $N { $(pub $f: $t),* }

        #[allow(non_upper_case_globals, dead_code)]
        impl $N {
            pub const NAMES: &'static [&'static str] = &[$(stringify!($f)),*];
            $(pub const $f: $crate::spatial::abi::Param =
                $crate::spatial::abi::Param(stringify!($f));)*
        }

        impl From<$N> for $crate::spatial::abi::Params {
            #[allow(unused_variables)]
            fn from(p: $N) -> Self {
                $crate::spatial::abi::Params {
                    items: vec![$((stringify!($f).to_string(), p.$f.into())),*],
                }
            }
        }
    };
}
pub(crate) use params;

/// An rcfg word layout: one `u32` field a word, in rcfg order (after the route table); `words()`
/// in that order and `FIELDS` their names.
macro_rules! words {
    ($(#[$m:meta])* $v:vis struct $N:ident { $($(#[$fm:meta])* $f:ident),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        $v struct $N { $($(#[$fm])* pub $f: u32),* }

        #[allow(dead_code)]
        impl $N {
            pub const FIELDS: &'static [&'static str] = &[$(stringify!($f)),*];

            pub fn words(&self) -> Vec<u32> {
                vec![$(self.$f),*]
            }
        }
    };
}
pub(crate) use words;
