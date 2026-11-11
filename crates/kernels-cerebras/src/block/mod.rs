//! Blocks: the unit a wafer program is composed of. A block is one part of the model's
//! dataflow as it sits on the wafer (the units and their chains, a segment kind, the head, the
//! token loop, a storage column ...): it places its PEs, lays its links, declares the colours of
//! the classes it owns, states per-PE param differences and copies its data. A family is a list
//! of blocks ([`super::family::Family::blocks`]); [`super::program::Program::build`] runs each
//! step over the list in order.
//!
//! A block owns its vocabulary: the [`Class`]es, [`super::wiring::Slot`]s and
//! [`super::wiring::NoteKey`]s it lays are consts in its module, and the kernels' rcfg writers
//! read the slots by them. Nothing in the core names a block's.

use super::abi::Overrides;
use super::floor::{Floor, FloorBuilder};
use super::plan::Plan;
use super::program::Program;
use super::weights::Copies;
use super::wiring::{Class, Wiring};

pub mod band;
pub mod gdn;
pub mod head;
pub mod head_groups;
pub mod moe;
pub mod ple;
pub mod row_runs;
pub mod segments;
pub mod shared_kv;
pub mod token_loop;
pub mod units;

type R<T> = Result<T, String>;

/// One part of a program. Every step is pure and defaults to nothing.
pub trait Block: Send + Sync {
    /// Places this block's PEs (the units first: later blocks place against them).
    fn floor(&self, _plan: &Plan, _b: &mut FloorBuilder) -> R<()> {
        Ok(())
    }

    /// Lays this block's links.
    fn wire(&self, _plan: &Plan, _floor: &Floor, _w: &mut Wiring) -> R<()> {
        Ok(())
    }

    /// Refines the floor once every link is laid (a block whose PEs are wherever links cross:
    /// Gemma's route-only PEs).
    fn settle(&self, _plan: &Plan, _w: &Wiring, _floor: &mut Floor) -> R<()> {
        Ok(())
    }

    /// The physical colours of the classes this block owns (a later block may recolour one).
    fn colours(&self) -> Vec<(Class, u8)> {
        Vec::new()
    }

    /// Per-PE kernel params over their kind's.
    fn overrides(&self, _prog: &Program, _o: &mut Overrides) -> R<()> {
        Ok(())
    }

    /// This block's copies onto the wafer, in placement order.
    fn bindings(&self, _prog: &Program, _out: &mut Copies) -> R<()> {
        Ok(())
    }
}
