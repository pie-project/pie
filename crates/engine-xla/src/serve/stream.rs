//! A fire's streamed weights (`crate::store::streamed`): the rows and the
//! expert places it reads, the staging it needs, and a cut fire's segments.

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::chunk::{WeightResidency, WeightUses};

use super::Shell;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::store::streamed::{Bank, Fire, Form, NOT_HELD, Streamed};

/// What `Shell::stream_fire` staged for a fire.
#[derive(Default)]
pub(super) struct Staged {
    pub fire: Option<Arc<Fire>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn stream_fire(
    streamed: &mut Streamed,
    pools: &mut crate::store::Pools,
    device: &crate::device::Device,
    handles: &crate::trace::Handles,
    inputs: &mut Inputs,
    tokens: &[i32],
    feeds: &[super::TokenFeed],
    feed_rows: &[(u32, usize)],
    real: &[bool],
    run: bool,
    stage: bool,
) -> Result<Staged> {
    if streamed.is_empty() {
        return Ok(Staged::default());
    }
    let mut ids: Vec<i32> = tokens.to_vec();
    if run
        && streamed
            .shapes
            .iter()
            .any(|s| matches!(s.form, Form::Rows { .. }))
    {
        // Ids the device writes: read them back (the feeding fire lands
        // first), so the host places their rows.
        for (feed, &(offset, lane)) in feeds.iter().zip(feed_rows) {
            debug_assert_eq!(feed.lane, lane);
            let words = feed.src.download()?;
            for (i, &word) in feed.words.iter().enumerate() {
                let at = word as usize * 4;
                let w = words.get(at..at + 4).ok_or_else(|| Fault::Program {
                    at: "serve::stream",
                    why: format!("a token feed names word {word} past its buffer"),
                })?;
                ids[offset as usize + i] = i32::from_le_bytes(w.try_into().expect("four bytes"));
            }
        }
    }
    let mut fire = Fire::default();
    for at in 0..streamed.shapes.len() {
        let shape = &streamed.shapes[at];
        match &shape.form {
            Form::Rows {
                param,
                block,
                vocab,
            } => {
                let (param, block, vocab) = (*param, *block, *vocab);
                let plane = shape.planes[0];
                let row = plane.row_bytes();
                let mut staged: BTreeMap<u32, u32> = BTreeMap::new();
                let mut idx = Vec::with_capacity(ids.len());
                let mut used = Vec::new();
                for (r, &id) in ids.iter().enumerate() {
                    let id = if id >= 0 && (id as u32) < vocab {
                        id as u32
                    } else {
                        0
                    };
                    let b = id / block;
                    let mut slot = shape.slots[b as usize];
                    if slot == NOT_HELD {
                        let next = staged.len();
                        slot = *staged.entry(b).or_insert_with(|| {
                            shape.staging.get(next).copied().unwrap_or(NOT_HELD)
                        });
                        if slot == NOT_HELD {
                            return Err(Fault::Residency(format!(
                                "a fire reads {} blocks of `{}` in no slot and the \
                                 residency names {} staging slots for them",
                                next + 1,
                                shape.name,
                                shape.staging.len()
                            )));
                        }
                    }
                    if real.get(r).copied().unwrap_or(false) {
                        used.push(b);
                    }
                    let off = pools.weight_offset(shape, slot)?;
                    idx.push(
                        i32::try_from(off / row + u64::from(id % block)).map_err(|_| {
                            Fault::Ceiling {
                                what: "rows of a weight view",
                                need: off / row,
                                have: i32::MAX as u64,
                            }
                        })?,
                    );
                }
                if run && !staged.is_empty() {
                    let fills: Vec<(u32, Vec<u8>)> = staged
                        .iter()
                        .map(|(&b, &slot)| Ok((slot, streamed.atom_bytes(at, b)?)))
                        .collect::<Result<_>>()?;
                    let shape = &streamed.shapes[at];
                    pools.write_weights(device, handles, shape, &fills)?;
                }
                if run {
                    streamed.count(at, used);
                }
                let view = pools
                    .weight_view(plane.width)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!("the {}-wide weight view", plane.width),
                    })?;
                let idx = inputs.i32s(handles, &idx, 1);
                fire.rows.push((param, view, idx));
            }
            Form::Experts { experts, layers } => {
                let experts = *experts;
                // The staging slots, a fixed count (the shape's floor) so the
                // program is the same whatever lies resident; past the ones
                // the runtime named, rows no expert in no slot is staged to.
                let floor = shape.floor as usize;
                for (layer_at, (_, banks)) in layers.iter().enumerate() {
                    let group = u32::try_from(fire.groups).unwrap_or(u32::MAX);
                    fire.groups += 1;
                    let slots = &shape.slots
                        [layer_at * experts as usize..(layer_at + 1) * experts as usize];
                    let missing = slots.iter().filter(|&&s| s == NOT_HELD).count();
                    fire.stages |= stage || missing > 0;
                    if missing > 0 && shape.staging.len() < floor.min(missing) {
                        return Err(Fault::Residency(format!(
                            "layer {layer_at} of `{}` holds {missing} experts in no slot and the \
                             residency names {} staging slots of the {floor} a fire may need",
                            shape.name,
                            shape.staging.len()
                        )));
                    }
                    for (plane, &param) in shape.planes.iter().zip(banks) {
                        let loc: Vec<i32> = slots
                            .iter()
                            .map(|&slot| expert_row(pools, &streamed.shapes[at], *plane, slot))
                            .collect::<Result<_>>()?;
                        let mut staging: Vec<i32> = shape
                            .staging
                            .iter()
                            .take(floor)
                            .map(|&slot| expert_row(pools, &streamed.shapes[at], *plane, slot))
                            .collect::<Result<_>>()?;
                        let pad = staging.last().copied().unwrap_or(0);
                        staging.resize(floor.max(1), pad);
                        let p = u32::try_from(param).unwrap_or(u32::MAX);
                        let host = pools.host_bank_root(p).ok_or_else(|| Fault::Unbound {
                            what: format!("the host copy of streamed bank {param}"),
                        })?;
                        let loc = inputs.i32s(handles, &loc, 1);
                        let staging = inputs.i32s(handles, &staging, 1);
                        fire.banks.push((
                            param,
                            Bank {
                                pool: pools.root(),
                                loc,
                                staging,
                                host: host.0,
                                plane_rows: host.1,
                                rows: plane.rows,
                                experts,
                                group,
                            },
                        ));
                    }
                }
            }
        }
    }
    Ok(Staged {
        fire: Some(Arc::new(fire)),
    })
}

impl Shell {
    /// The streamed weights' residency from the next fire on
    /// (`Engine::weight_residency`): fills the slots newly named.
    pub fn weight_residency(&mut self, residency: &WeightResidency) -> Result<()> {
        if self.streamed.is_empty() {
            return Ok(());
        }
        let fills = self.streamed.residency(residency)?;
        let mut by: BTreeMap<usize, Vec<(u32, Vec<u8>)>> = BTreeMap::new();
        for (at, atom, slot) in fills {
            by.entry(at)
                .or_default()
                .push((slot, self.streamed.atom_bytes(at, atom)?));
        }
        for (at, slots) in by {
            let shape = &self.streamed.shapes[at];
            self.pools
                .write_weights(&self.device, &self.handles, shape, &slots)?;
        }
        self.streamed.settle_held();
        Ok(())
    }

    /// The reads of each streamed atom since the last call
    /// (`Engine::weight_uses`), and where each shape's atoms lie.
    pub fn weight_uses(&mut self) -> WeightUses {
        let (uses, held) = self.streamed.uses();
        WeightUses { uses, held }
    }

    /// The streamed weights' shapes, for the load's facts.
    #[must_use]
    pub fn weight_shapes(&self) -> Vec<engine::chunk::WeightShape> {
        self.streamed.facts()
    }
}

/// The pool row `plane` of an expert in `slot` starts at; -1 for an expert
/// in no slot (a fire stages it: `kernels_xla::hlo::Func::stage_misses`).
fn expert_row(
    pools: &crate::store::Pools,
    shape: &crate::store::streamed::Shape,
    plane: crate::store::streamed::Plane,
    slot: u32,
) -> Result<i32> {
    if slot == NOT_HELD {
        return Ok(-1);
    }
    let off = pools.weight_offset(shape, slot)? + plane.offset;
    i32::try_from(off / plane.view_row_bytes()).map_err(|_| Fault::Ceiling {
        what: "rows of a weight view",
        need: off / plane.view_row_bytes(),
        have: i32::MAX as u64,
    })
}
