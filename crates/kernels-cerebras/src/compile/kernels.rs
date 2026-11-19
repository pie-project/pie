//! The wafer program kernels: one [`super::abi::Kernel`] descriptor a CSL source, its params
//! struct and its rcfg writer, which sees one PE as a [`Pe`]. The program's kernels ([`of`]) are
//! read off its elements: the units' role kernels by [`KernelSet`], the head's by where the
//! floorplan puts it ([`head`]), the side input's store kernel, and the relay for gap slots.

use serde::{Deserialize, Serialize};

use super::abi::Kernels;
use super::compose::Needs;
use super::fabric::{ChainMember, Colour, Endpoint, PeIo, RootQueues};
use super::floor::Coord;
use super::floorplan::{Aux, HeadAt};
use super::layers;
use super::plan::Shape;
use super::program::Program;
use super::role::PeRole;
use super::wiring::{Class, NoteKey, Slot};

pub mod common;
pub mod head;
pub mod runs;
pub mod snake;
pub mod split;
pub mod stacked;

type R<T> = Result<T, String>;

/// The role kernels a program's units run: one CSL ABI a set (the kernels' params, rcfg words,
/// exported symbols, the copies they read and the cut that makes them), each the mixers and norms
/// of a family of layer programs. Chosen by what the layer program's mixers need
/// ([`KernelSet::of`]), not by where the units lie: the placement is the floorplan's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelSet {
    /// Attention over a layer's own cache and the gated delta rule, direct norms (`MIXK 0`:
    /// Qwen3.5, Qwen3, Llama 3).
    Own,
    /// Sliding / global caches, caches read across layers and a per-token side input, deferred
    /// norms (`MIXK 1`, `PLE`, `KVS`: Gemma 4 E2B / E4B).
    Shared,
    /// Sliding / global caches over attention and plain row runs, deferred norms (`MIXK 1`,
    /// `AROW`, `KEV`: Gemma 4 12B).
    Windows,
    /// Routed experts: attention | MoE halves (`MIXK 2`, `3`: gpt-oss, Qwen3 MoE).
    Experts,
}

impl KernelSet {
    /// The set's name (the plan's record, error messages).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            KernelSet::Own => "own",
            KernelSet::Shared => "shared",
            KernelSet::Windows => "windows",
            KernelSet::Experts => "experts",
        }
    }

    /// The set that runs layers that need `needs`: routed experts the halves'; a side input or a
    /// cache read across layers the shared caches'; windows the windowed rows'; else the own
    /// caches'.
    #[must_use]
    pub fn of(needs: &Needs) -> KernelSet {
        if needs.experts {
            KernelSet::Experts
        } else if needs.side_in || needs.shared_cache {
            KernelSet::Shared
        } else if needs.sliding {
            KernelSet::Windows
        } else {
            KernelSet::Own
        }
    }

    /// The set `shape`'s layers take.
    #[must_use]
    pub fn of_shape(shape: &Shape) -> KernelSet {
        KernelSet::of(&Needs::of(&layers::of(shape)))
    }
}

/// Each kind's kernel and its params for `prog` (levers applied by
/// [`super::emit::params::kernels`]).
pub fn of(prog: &Program) -> R<Kernels> {
    let mut out = match prog.roles {
        KernelSet::Own => snake::kernels(prog),
        KernelSet::Shared => stacked::kernels(prog),
        KernelSet::Windows => runs::kernels(prog),
        KernelSet::Experts => split::kernels(prog),
    }?;
    let fp = &prog.floorplan;
    let store = fp.aux.iter().any(|a| matches!(a, Aux::Store(_)));
    out.append(&mut match fp.head.at {
        HeadAt::East => head::east(prog, store)?,
        HeadAt::Groups => head::groups(prog)?,
        HeadAt::Band => split::band(prog)?,
    });
    if store {
        out.append(&mut stacked::store(prog)?);
    }
    if !prog.floor.relays.is_empty() {
        out.insert(
            super::role::Kind::RELAY,
            super::abi::KernelBuild::of(&common::RELAY, super::abi::Params::default())?,
        );
    }
    Ok(out)
}

/// Every name the kernels of `set` export, with its element type, in export order: the ABI
/// between the kernels and the host's memcpy.
#[must_use]
pub fn symbols(set: KernelSet) -> &'static [(&'static str, &'static str)] {
    match set {
        KernelSet::Own => snake::SYMBOLS,
        KernelSet::Shared => stacked::SYMBOLS,
        KernelSet::Windows => runs::SYMBOLS,
        KernelSet::Experts => split::SYMBOLS,
    }
}

/// The bits of a unit's layer mask (`KM`: a `u16` in `cy.csl` and `weight.csl`).
pub const KM_BITS: u32 = 16;

/// A unit's layer mask over its `k` layers (bit `l` where `of(l)`), or why a unit of `k` layers
/// has none.
pub fn layer_mask(k: u32, of: impl Fn(u32) -> bool) -> R<u32> {
    if k > KM_BITS {
        return Err(format!(
            "{k} layers a unit past the layer mask's {KM_BITS} bits"
        ));
    }
    Ok((0..k).filter(|&l| of(l)).fold(0, |m, l| m | (1 << l)))
}

/// A colour as an rcfg word.
pub fn word(c: Colour) -> u32 {
    u32::from(c.0)
}

/// `a`, unless absent or colour 0, else `b` (the tools' `a or b`).
pub fn or(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match a {
        Some(v) if v != 0 => Some(v),
        _ => b,
    }
}

/// One PE as its rcfg writer sees it: its bindings and notes, the program's palette, and a
/// colour it must never park on and counts as final (`reserved`: gpt-oss's band).
pub struct Pe<'a> {
    pub prog: &'a Program,
    pub at: Coord,
    pub io: &'a PeIo,
    pub reserved: Option<u8>,
}

impl<'a> Pe<'a> {
    pub fn new(prog: &'a Program, at: Coord) -> Pe<'a> {
        Pe {
            prog,
            at,
            io: prog.fabric.io(at),
            reserved: None,
        }
    }

    pub fn role(&self) -> &'a PeRole {
        self.prog.floor.roles.get(self.at)
    }

    /// The palette's colour of `k`.
    pub fn c(&self, k: Class) -> R<u32> {
        self.prog.palette.get(k).map(word)
    }

    /// The OR of the notes keyed `k` at this PE (0: none).
    pub fn note(&self, k: NoteKey) -> u32 {
        self.io.notes.get(&k).copied().unwrap_or(0)
    }

    pub fn member(&self, s: Slot) -> R<ChainMember> {
        match self.io.slots.get(&s) {
            Some(Endpoint::Member(m)) => Ok(*m),
            e => Err(format!(
                "{:?}: {} is {e:?}, not a chain member",
                self.at, s.name
            )),
        }
    }

    pub fn root(&self, s: Slot) -> R<RootQueues> {
        match self.io.slots.get(&s) {
            Some(Endpoint::Root(r)) => Ok(*r),
            e => Err(format!(
                "{:?}: {} is {e:?}, not a chain root",
                self.at, s.name
            )),
        }
    }

    /// A root's two halves, both required.
    pub fn halves(&self, s: Slot) -> R<(u32, u32)> {
        let r = self.root(s)?;
        let half = |h: Option<Colour>, n| {
            h.map(word)
                .ok_or_else(|| format!("{:?}: {} root without a {n} half", self.at, s.name))
        };
        Ok((half(r.lo, "low")?, half(r.hi, "high")?))
    }

    /// The colour a path / multicast endpoint left in `s`.
    pub fn colour(&self, s: Slot) -> Option<u32> {
        match self.io.slots.get(&s)? {
            Endpoint::Send(c) | Endpoint::Recv(c) => Some(word(*c)),
            _ => None,
        }
    }

    /// A relay's turn (from, to) in `s`.
    pub fn turn(&self, s: Slot) -> Option<(u32, u32)> {
        match self.io.slots.get(&s)? {
            Endpoint::Turn(a, b) => Some((word(*a), word(*b))),
            _ => None,
        }
    }

    /// The route table, `slots` slots.
    pub fn routes(&self, slots: usize) -> R<Vec<u32>> {
        self.prog.fabric.route_words(self.at, slots)
    }

    /// `n` colours to park unused queues on.
    pub fn parks(&self, n: usize) -> R<Vec<u32>> {
        let avoid: Vec<Colour> = self.reserved.map(Colour).into_iter().collect();
        Ok(self
            .prog
            .fabric
            .parks_avoiding(self.at, n, &avoid)?
            .into_iter()
            .map(word)
            .collect())
    }

    /// A spare colour for the kernel's temporary route.
    pub fn spare(&self, finals: &[u32], comptime: &[u8]) -> R<u32> {
        let mut f: Vec<Colour> = finals.iter().map(|&c| Colour(c as u8)).collect();
        f.extend(self.reserved.map(Colour));
        let ct: Vec<Colour> = comptime.iter().map(|&c| Colour(c)).collect();
        self.prog.fabric.spare(self.at, &f, &ct).map(word)
    }
}
