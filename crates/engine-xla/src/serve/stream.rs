//! A fire's streamed weights (`crate::store::streamed`): the rows and the
//! expert places it reads, the staging it needs, and a cut fire's segments.

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::chunk::{WeightResidency, WeightUses};

use super::Shell;
use crate::error::{Fault, Result};
use crate::inputs::Inputs;
use crate::pjrt::Buffer;
use crate::store::streamed::{Bank, Fire, Form, NOT_HELD, Streamed};

/// A cut fire's segments before its last program: each program and the
/// activation roots it returns, and the cut each closes on.
pub(super) struct Segmented {
    pub segments: Vec<(std::sync::Arc<crate::device::Program>, Vec<u32>)>,
    pub cuts: Vec<crate::store::streamed::Cut>,
}

/// What `Shell::stream_fire` staged for a fire.
#[derive(Default)]
pub(super) struct Staged {
    pub fire: Option<Arc<Fire>>,
    /// Where each streamed bank's expert places start in the pack.
    pub locs: Vec<(usize, u32)>,
    /// What the program key names: which layers the fire cuts at.
    pub key: String,
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
    let mut locs = Vec::new();
    let mut key = String::new();
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
                for (layer_at, (_, banks)) in layers.iter().enumerate() {
                    let whole = streamed.layer_whole(at, layer_at);
                    if !whole {
                        key.push_str(&format!("{at}.{layer_at},"));
                    }
                    let slots = &shape.slots
                        [layer_at * experts as usize..(layer_at + 1) * experts as usize];
                    for (plane, &param) in shape.planes.iter().zip(banks) {
                        let loc: Vec<i32> = slots
                            .iter()
                            .map(|&slot| expert_row(pools, &streamed.shapes[at], *plane, slot))
                            .collect::<Result<_>>()?;
                        let view = pools
                            .weight_view(plane.view)
                            .ok_or_else(|| Fault::Unbound {
                                what: format!("the {}-wide weight view", plane.view),
                            })?;
                        locs.push((param, inputs.pack_offset()));
                        let loc = inputs.i32s(handles, &loc, 1);
                        fire.banks.push((
                            param,
                            Bank {
                                view,
                                loc,
                                rows: plane.rows,
                                view_rows: plane.view_rows(),
                                experts,
                                shape: at,
                                layer: layer_at,
                                whole,
                            },
                        ));
                    }
                }
            }
        }
    }
    Ok(Staged {
        fire: Some(Arc::new(fire)),
        locs,
        key,
    })
}

impl Shell {
    /// Lays this fire's streamed weights: each streamed table's row for
    /// every token row (staging the blocks in no slot, unless `run` is
    /// false: a fire traced and not run), each streamed bank's expert
    /// places. `tokens` are the fire's token rows, `feeds` the lanes whose
    /// tokens the device supplies (read back here: the host places their
    /// rows), `real` the rows of real lanes.
    /// Runs a cut fire's segments before its last program: after each, the
    /// routes of the layer it cut at come back, the experts they name in no
    /// slot are staged and the layer's banks re-placed in `inputs`. Answers
    /// the activations the last program reads.
    pub(super) fn run_segments(
        &mut self,
        segmented: &Segmented,
        inputs: &mut Inputs,
        staged: &Staged,
    ) -> Result<BTreeMap<u32, Buffer>> {
        let mut carried: BTreeMap<u32, Buffer> = BTreeMap::new();
        for ((program, roots), cut) in segmented.segments.iter().zip(&segmented.cuts) {
            let outs = crate::exec::run_carried(
                &self.device,
                program,
                Some(&self.weights),
                &mut self.pools,
                inputs,
                &carried,
                true,
            )?;
            if outs.len() != roots.len() {
                return Err(Fault::Unbound {
                    what: format!(
                        "a fire segment returned {} activations for the {} it carries",
                        outs.len(),
                        roots.len()
                    ),
                });
            }
            for (&root, out) in roots.iter().zip(outs) {
                carried.insert(root, out);
            }
            // The routes, as the segment landed them.
            let (root, skip) =
                self.handles
                    .locate(cut.routes.buf)
                    .ok_or_else(|| Fault::Unbound {
                        what: "the handle of a cut's routes".to_string(),
                    })?;
            let buffer = carried.get(&root).ok_or_else(|| Fault::Unbound {
                what: "the routes of a cut, which its segment did not return".to_string(),
            })?;
            let bytes = buffer.download()?;
            let width = cut.routes.width as usize;
            let from = skip as usize * width * 4;
            let to = from + cut.routes.rows as usize * width * 4;
            let words = bytes.get(from..to).ok_or_else(|| Fault::Unbound {
                what: "the rows of a cut's routes".to_string(),
            })?;
            let shape = self.streamed.shapes[cut.shape].clone();
            let Form::Experts {
                experts,
                ref layers,
            } = shape.form
            else {
                continue;
            };
            let mut named: Vec<u32> = words
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| i32::from_le_bytes(*w))
                .filter(|&e| e >= 0 && (e as u32) < experts)
                .map(|e| e as u32)
                .collect();
            named.sort_unstable();
            named.dedup();
            let base = cut.layer as u32 * experts;
            self.streamed
                .count(cut.shape, named.iter().map(|&e| base + e));
            let misses: Vec<u32> = named
                .iter()
                .copied()
                .filter(|&e| shape.slots[(base + e) as usize] == NOT_HELD)
                .collect();
            if misses.len() > shape.staging.len() {
                return Err(Fault::Residency(format!(
                    "a fire's routes read {} experts of layer {} of `{}` in no slot and the \
                     residency names {} staging slots for them",
                    misses.len(),
                    cut.layer,
                    shape.name,
                    shape.staging.len()
                )));
            }
            let mut fills = Vec::with_capacity(misses.len());
            let mut placed = BTreeMap::new();
            for (&e, &slot) in misses.iter().zip(&shape.staging) {
                fills.push((slot, self.streamed.atom_bytes(cut.shape, base + e)?));
                placed.insert(e, slot);
            }
            self.pools
                .write_weights(&self.device, &self.handles, &shape, &fills)?;
            for (&plane, &param) in shape.planes.iter().zip(&layers[cut.layer].1) {
                let Some(&(_, offset)) = staged.locs.iter().find(|(p, _)| *p == param) else {
                    continue;
                };
                let slots = &shape.slots[base as usize..(base + experts) as usize];
                let loc: Vec<i32> = (0..experts)
                    .map(|e| {
                        let slot = placed.get(&e).copied().unwrap_or(slots[e as usize]);
                        expert_row(&self.pools, &shape, plane, slot)
                    })
                    .collect::<Result<_>>()?;
                inputs.set_i32s(offset, &loc);
            }
        }
        Ok(carried)
    }

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

/// The row of `plane`'s view an expert in `slot` starts at; an expert
/// in no slot reads the null chunk (zeros) until a cut stages it.
fn expert_row(
    pools: &crate::store::Pools,
    shape: &crate::store::streamed::Shape,
    plane: crate::store::streamed::Plane,
    slot: u32,
) -> Result<i32> {
    if slot == NOT_HELD {
        return Ok(0);
    }
    let off = pools.weight_offset(shape, slot)? + plane.offset;
    i32::try_from(off / plane.view_row_bytes()).map_err(|_| Fault::Ceiling {
        what: "rows of a weight view",
        need: off / plane.view_row_bytes(),
        have: i32::MAX as u64,
    })
}
