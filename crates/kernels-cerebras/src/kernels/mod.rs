//! The wafer program kernels: one [`super::abi::Kernel`] descriptor a CSL source, its params
//! struct and its rcfg writer. An rcfg writer sees one PE as a [`Pe`]: the queues the blocks'
//! links bound there (by slot), the notes they left, the program's palette and fabric. Each
//! layout's role kernels take their params from the plan ([`of`]) and export their symbols in one
//! order ([`symbols`]):
//!
//! - [`common`]: the word layouts every layout's weight, CX, CY, head and head control kernels
//!   share (their leading words), with the codes the kernels read named;
//! - [`snake`]: the unit snake's (`weight.csl`, `cx.csl`, `cy.csl`, `head_pe.csl`,
//!   `head_ctl.csl`, `relay.csl`);
//! - [`stacked`]: the stacked columns' (`weight.csl`, `cx.csl`, `cy.csl`, `head_ctl.csl` (TB 1),
//!   `ple_gemma.csl`; `head_pe.csl`);
//! - [`runs`]: the row runs' (`weight.csl` with row roles, `cx.csl`, `cy.csl`, `head_pe.csl`
//!   with head groups, `head_ctl.csl`);
//! - [`split`]: the attention / MoE halves' (`weight.csl`, `cx.csl`, `cy.csl`'s halves,
//!   `head_pe_gptoss.csl` on the band, `head_ctl.csl`).

use super::abi::Kernels;
use super::compose::Layout;
use super::fabric::{ChainMember, Colour, Endpoint, PeIo, RootQueues};
use super::floor::Coord;
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

/// Each kind's kernel and its params for `prog`, by its layout (levers applied by
/// [`super::emit::params::kernels`]).
pub fn of(prog: &Program) -> R<Kernels> {
    match prog.layout {
        Layout::Snake => snake::kernels(prog),
        Layout::Stacked => stacked::kernels(prog),
        Layout::Runs => runs::kernels(prog),
        Layout::Split => split::kernels(prog),
    }
}

/// Every name the kernels of `layout` export, with its element type, in export order: the ABI
/// between the kernels and the host's memcpy.
#[must_use]
pub fn symbols(layout: Layout) -> &'static [(&'static str, &'static str)] {
    match layout {
        Layout::Snake => snake::SYMBOLS,
        Layout::Stacked => stacked::SYMBOLS,
        Layout::Runs => runs::SYMBOLS,
        Layout::Split => split::SYMBOLS,
    }
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
