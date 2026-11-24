//! Beam search's per-beam page lists (`state-geometry.parents`).
//!
//! The beams share one working set: the prompt is folded into it, and the
//! pages of its reserve past its frozen full ones are a pool each beam holds
//! its own pages of. A beam's rows read the prompt and its own pages only,
//! maskless, to its own slot.
//!
//! The guest states one thing a fire: each beam's parent. The engine keeps
//! every beam's list and the free pages as state of its own and, before each
//! fire reads or appends, reorders the lists by the parents: a child shares
//! its parent's full pages; a parent's first child keeps its tail page and
//! every other child has the tail's slots copied into a free page; a page no
//! list names any more is handed out again a few fires later. Every page a
//! list names lies in the group's own working set's reserve, checked on the
//! device, a lane faulting by name where one does not.

use crate::eta::{Channel, ForwardPass, StateGeometry, WorkingSet, kv_page_size};
use eta_dsl::dtype;
use eta_dsl::value::Tensor;

/// `beams` lanes searching over one working set from `folded` frozen slots.
pub struct BeamGroup {
    beams: u32,
    reserve: u32,
    folded: u32,
    steps: u32,
    positions: Channel,
    slots: Channel,
    parents: Channel,
}

impl BeamGroup {
    /// Starts `beams` beams decoding `steps` fires from `folded` frozen
    /// slots, the first writing slot `folded`: `ws`
    /// reserves what the first column needs (each beam's page, the pages a
    /// fire hands out, those freed in the frames in flight, and the prompt's
    /// own page where it ends mid-page); the runtime grows the reservation a
    /// column at a time before the fire that opens it.
    pub fn new(ws: &WorkingSet, beams: u32, folded: u32, steps: u32) -> Result<BeamGroup, String> {
        if beams == 0 || steps == 0 {
            return Err("a beam group decodes at least one beam one step".to_string());
        }
        let page = kv_page_size().max(1);
        let f0 = folded / page;
        folded
            .checked_add(steps - 1)
            .ok_or("the beams' slots pass the u32 range")?;
        // The frames the engine keeps in flight: a page freed by a fire is
        // handed out again only that many fires later (§3, §6.3).
        let frames = crate::model::frames_in_flight().max(1);
        let reserve = beams
            .checked_mul(2)
            .and_then(|pages| pages.checked_add(beams.checked_mul(frames)?))
            .and_then(|pages| pages.checked_add(1))
            .ok_or("the beams' pages pass the u32 range")?;
        let end = (f0 + reserve)
            .checked_mul(page)
            .ok_or("the beams' reserve passes the u32 range")?;
        ws.reserve(end)?;
        Ok(BeamGroup {
            beams,
            reserve,
            folded,
            steps,
            positions: Channel::from(vec![folded; beams as usize]),
            slots: Channel::from(vec![folded; beams as usize]),
            parents: Channel::from((0..beams).collect::<Vec<u32>>()),
        })
    }

    /// Where the beams' rows land and what they read: each beam's slot (its
    /// depth, every beam's the same), its position, and its parent.
    #[must_use]
    pub fn geometry(&self) -> StateGeometry<'_> {
        StateGeometry {
            positions: Some(&self.positions),
            slots: Some(&self.slots),
            parents: Some(&self.parents),
            // The search's depth: its folded slots and a slot a fire.
            parents_end: Some(self.folded + self.steps),
            ..StateGeometry::default()
        }
    }

    /// In the epilogue: each beam continues `parent[beam]`'s hypothesis
    /// (`parent` a `[beams]` tensor of the beams this fire read). Puts the
    /// parents the next fire reorders by, and its slots and positions.
    pub fn reorder(&self, parent: &Tensor) {
        let _ = self.parents.take();
        self.parents.put(eta_dsl::value::cast(parent, dtype::u32));
        let slot = self.slots.take();
        self.slots.put(&slot + 1u32);
        let position = self.positions.take();
        self.positions.put(&position + 1u32);
    }

    /// Ends the search on beam `beam` after `fires` fires of `pass` (the
    /// pass the group's geometry is bound to), once the last has landed:
    /// the working set then holds that beam's hypothesis at slots
    /// `[folded, folded + fires)` in its own order, its recurrent rows too
    /// on a hybrid, and later passes bind it as any set; the other beams'
    /// pages are let go. On a hybrid its rows
    /// are folded into the state by an ordinary pass that folds them.
    pub fn finish(&self, pass: &ForwardPass, beam: u32, fires: u32) -> Result<(), String> {
        if beam >= self.beams {
            return Err(format!("beam {beam} of a group of {}", self.beams));
        }
        let end = self
            .folded
            .checked_add(fires)
            .ok_or("the beams' slots pass the u32 range")?;
        pass.adopt_beam(beam, end)
    }

    /// The beams.
    #[must_use]
    pub fn beams(&self) -> u32 {
        self.beams
    }

    /// The pool pages the group reserved for its first column.
    #[must_use]
    pub fn reserve(&self) -> u32 {
        self.reserve
    }
}
