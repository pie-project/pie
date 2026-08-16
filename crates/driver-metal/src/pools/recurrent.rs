//! The recurrent state pool: two planes per linear-attention layer.
//!
//! [`layout::recurrent::Shape`](crate::layout::recurrent::Shape) says how big;
//! this allocates it. The split is the same one `pools::kv` draws and for the
//! same reason — the arithmetic is correct with no device, the memory is not.
//!
//! # What it is not
//!
//! It is not a pager. A KV pool hands out pages by the token and moves them
//! between requests; these planes hand out a SEAT, which a request holds from
//! its first token to its last and nobody else touches meanwhile. So there is
//! no move plan, no elastic growth, no reservation arithmetic: allocate once
//! at load, zero it, and index it by slot forever after.
//!
//! It is also not two pools despite being two planes. `conv` and `state` are
//! allocated and zeroed together because they are the same fact — a request's
//! linear-attention memory — split only by the two shapes the kernels want,
//! and a build that had one without the other would run and be wrong.

use crate::device::{Allocation, Context, Regions};
use crate::error::Result;
use crate::layout::recurrent::Shape;
use crate::layout::region::Region as _;

/// One linear-attention layer's three planes.
#[derive(Debug)]
pub struct Layer {
    /// The conv windows a fire READS — one per slot, and canonical between
    /// fires.
    pub conv: Allocation,
    /// The conv windows a fire WRITES. Same size, same indexing; carried back
    /// into [`Self::conv`] by [`Pool::carry_forward`] once the fire retires.
    pub new_conv: Allocation,
    /// One DeltaNet memory matrix per slot, updated in place.
    pub state: Allocation,
}

/// Every linear-attention layer's planes, in stack order.
#[derive(Debug)]
pub struct Pool {
    shape: Shape,
    layers: Vec<Layer>,
}

impl Pool {
    /// Allocate and zero a pool at `shape`.
    ///
    /// Zeroing is not hygiene. A fresh slot's conv window IS its history —
    /// the taps before a prompt's first token are defined to be zero — and
    /// the DeltaNet memory starts empty by the same definition, so the zero
    /// here is the model's initial condition and not a courtesy.
    ///
    /// # Errors
    ///
    /// Any layer's allocation. Nothing partial survives: the vector is local
    /// until every layer is in it.
    pub fn allocate(context: &Context, shape: Shape) -> Result<Self> {
        let mut layers = Vec::with_capacity(shape.linear_layers as usize);
        for _ in 0..shape.linear_layers {
            let conv = Allocation::new(context, shape.conv_bytes_per_layer(), "gdn conv state")?;
            let new_conv =
                Allocation::new(context, shape.conv_bytes_per_layer(), "gdn conv scratch")?;
            let state = Allocation::new(context, shape.state_bytes_per_layer(), "gdn recurrent")?;
            // SAFETY: freshly allocated; nothing is encoded against any.
            unsafe {
                conv.zero(0, conv.len())?;
                new_conv.zero(0, new_conv.len())?;
                state.zero(0, state.len())?;
            }
            layers.push(Layer { conv, new_conv, state });
        }
        Ok(Self { shape, layers })
    }

    /// The shape it was allocated at.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// The `l`th LINEAR layer's planes — counted over linear layers only, not
    /// over the stack.
    ///
    /// A hybrid's third layer may be its first linear one; the caller that
    /// walks the stack owns that mapping, because the pool never learned
    /// which stack positions it was allocated for and would have to be told
    /// twice to answer in stack coordinates.
    #[must_use]
    pub fn layer(&self, l: u32) -> Option<&Layer> {
        self.layers.get(l as usize)
    }

    /// How many layers it holds.
    #[must_use]
    pub fn layers(&self) -> u32 {
        u32::try_from(self.layers.len()).unwrap_or(u32::MAX)
    }

    /// Total bytes across every plane of every layer.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.shape.total_bytes()
    }

    /// Name every plane to the residency set, so a fire may read them.
    pub fn register(&self, regions: &mut Regions) {
        for l in &self.layers {
            regions.add(&l.conv);
            regions.add(&l.new_conv);
            regions.add(&l.state);
        }
    }

    /// Zero one seat's conv pair and DeltaNet memory in every layer.
    ///
    /// What a request STARTING means: the taps before its first token are
    /// defined to be zero and its memory is empty, so a seat handed on from a
    /// finished request has to be cleared before the new one reads it. Both
    /// Both conv planes go, not just the read one: the carry back after the
    /// next fire copies the whole write plane over the read one, so a stale
    /// window left in the write plane would come back a step later.
    ///
    /// # Errors
    ///
    /// A slot past the pool's count, which is a scheduler admitting beyond
    /// what `rs_cache_slots` advertised.
    pub fn clear_slot(&self, slot: u32) -> Result<()> {
        let shape = self.shape;
        for l in &self.layers {
            // SAFETY: called between fires, from the launch path, before
            // anything is encoded against this seat.
            unsafe {
                l.conv
                    .zero(shape.conv_offset(slot), shape.conv_bytes_per_slot())?;
                l.new_conv
                    .zero(shape.conv_offset(slot), shape.conv_bytes_per_slot())?;
                l.state
                    .zero(shape.state_offset(slot), shape.state_bytes_per_slot())?;
            }
        }
        Ok(())
    }

    /// Make what the last fire wrote what the next fire reads.
    ///
    /// The whole plane and not the fired slots. That is the correctness
    /// argument, not an optimisation: the two planes are identical for every
    /// untouched slot at the start of a fire, the fire changes only the slots
    /// it names, so copying everything leaves them identical again. Copying
    /// only the named ones would leave the write plane one step behind for
    /// everyone else and the next carry back would serve them that.
    ///
    /// Called after the fire has RETIRED. It is a `memmove` over shared
    /// memory rather than a blit, so nothing orders it against the GPU except
    /// the caller having waited.
    ///
    /// # Errors
    ///
    /// A plane that is not host addressable, which these are.
    ///
    /// # Safety
    ///
    /// No fire may still be reading or writing the planes.
    pub unsafe fn carry_forward(&self) -> Result<()> {
        for l in &self.layers {
            // SAFETY: the caller has waited; the two planes are the same size
            // and are distinct allocations.
            unsafe { l.conv.copy(0, l.new_conv.handle(), 0, l.new_conv.len())? };
        }
        Ok(())
    }

    /// Zero every plane again — a reset of every slot at once.
    ///
    /// # Errors
    ///
    /// A plane that is not host addressable, which these are.
    ///
    /// # Safety
    ///
    /// No fire may be reading the pool. The caller owns that, as it does for
    /// every other write to a bound address.
    pub unsafe fn zero_all(&self) -> Result<()> {
        for l in &self.layers {
            unsafe {
                l.conv.zero(0, l.conv.len())?;
                l.new_conv.zero(0, l.new_conv.len())?;
                l.state.zero(0, l.state.len())?;
            }
        }
        Ok(())
    }
}
