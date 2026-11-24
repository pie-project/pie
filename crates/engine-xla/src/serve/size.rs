//! The stage-only pass: what a load's fires stage, measured by staging the
//! widest fire of each token rung and lane count through the same code real
//! fires take (`Shell::fire_inner`, stopped once its inputs are staged,
//! before anything is traced). Each rung's largest set is what its fires
//! take of the pool's slab (`ChunkFacts::act_chunks`); the largest of all
//! is the slab and every staging-ring slot.
//!
//! Each synthetic fire is the worst case of every array it stages:
//! - per-row tables (tokens, positions, cache rows, owning request, adapter
//!   routes, write pages and offsets): the rung's whole rows;
//! - per-lane tables (kv geometry, record ids): the lane count, every lane
//!   seated;
//! - page tables: every lane at the page bound (its slot's every page);
//! - readout tables: every row of every lane read back;
//! - the custom mask plane (classes that run one): every lane states a
//!   mask over its whole context, so the plane is rows by the widest
//!   stride a lane's keys can reach;
//! - device-fed token tables: every lane's every row fed off the device;
//! - recurrent replay maps (a load with recurrent state): every lane
//!   buffers its rows and the fire replays its rung's carried room
//!   (`crate::rs::carried_room`), spread over its lanes;
//! - streamed-expert staging tables (a load streaming experts): every fire
//!   staged as one whose experts are in no slot (`Shell::stage_all`);
//! - guest input planes: every declared input fed on every lane, a plane
//!   of its whole extent (the fire's lanes or rows by its width).
//!
//! The classes enumerated are the words the load's compiled classes run
//! (decode, prefill, and their masked forms where a class runs a mask),
//! not the warm ladder's: the ladder holds no masked, wide or corrected
//! fire. A real fire past its slab is refused by name
//! (`crate::store::stage::Ring::write`), the proof none does.

use engine::fire::{FoldLen, Mask, Masking, RsVerb};
use poem_ir::Request;

use super::{Lane, Seated, Shell, TokenFeed};
use crate::error::Result;

impl Shell {
    /// `(rung tokens, rows)` each rung's widest fire stages, ascending.
    pub(crate) fn staged_sets(&mut self) -> Result<Vec<(u32, u32)>> {
        let facts = self.trace().facts.clone();
        let classes = &self.compiled.classes;
        let masked = !self.masked.is_empty();
        let mut words = vec![
            (facts.word(&Request::new(1, false)), true, false),
            (facts.word(&Request::new(2, false)), false, false),
        ];
        if masked {
            words.push((facts.word(&Request::new(1, true)), true, true));
            words.push((facts.word(&Request::new(2, true)), false, true));
        }
        // Only words a compiled class runs.
        words.retain(|&(word, ..)| classes.class_of(word & classes.mask).is_some());
        let buckets = self.budgets.tokens.buckets.clone();
        let lanes_cap = self.budgets.tokens.max_lanes.min(self.held.len() as u32).max(1);
        let page = self.paging().page_size.max(1);
        let pages = self.paging().pages_per_slot.max(1);
        let context = pages * page;
        let mut sets: Vec<(u32, u32)> = Vec::new();
        self.sizing = Some(0);
        self.stage_all = self.streams_experts();
        let mut out = Ok(());
        'rungs: for &rung in &buckets {
            let mut most = 0u32;
            for &(word, decode, with_mask) in &words {
                // A decode fire's rows are its lanes; a prefill's the
                // rung's, over every lane count it can take.
                let counts: Vec<u32> = if decode {
                    if rung > lanes_cap {
                        continue;
                    }
                    vec![rung]
                } else {
                    // A masked load's mixed fire (prompts beside decodes)
                    // stages the mask plane over as many lanes as the rung
                    // seats, past a prefill's (`Shell::fire_full`): the
                    // rung's rows over its most lanes, each at the page
                    // bound, bound it.
                    let mixed = with_mask.then(|| rung.min(lanes_cap));
                    buckets
                        .iter()
                        .copied()
                        .filter(|&n| n <= lanes_cap && n * 2 <= rung)
                        .chain(std::iter::once(1))
                        .chain(mixed)
                        .collect()
                };
                for lanes in counts {
                    match self.stage_widest(word, rung, lanes, decode, with_mask, context) {
                        Ok(rows) => most = most.max(rows),
                        Err(fault) => {
                            out = Err(fault);
                            break 'rungs;
                        }
                    }
                }
            }
            if most > 0 {
                sets.push((rung, most));
            }
        }
        self.sizing = None;
        self.stage_all = false;
        out.map(|()| sets)
    }

    /// The rows the widest fire of `lanes` lanes at `rung` rows stages.
    fn stage_widest(
        &mut self,
        word: u64,
        rung: u32,
        lanes: u32,
        decode: bool,
        with_mask: bool,
        context: u32,
    ) -> Result<u32> {
        let slots = self.held.len().max(1) as u32;
        let page = self.paging().page_size.max(1);
        let rows: Vec<u32> = (0..lanes)
            .map(|at| {
                if decode {
                    1
                } else {
                    rung / lanes + u32::from(at < rung % lanes)
                }
            })
            .collect();
        // The most rows a fire reads out, the store's (`readout_store_rows`),
        // shared over the lanes: each reads out its last rows up to its
        // share, so the widest fire reads out what a served one can and no
        // more (it once read every row, past the ring). A load with no store
        // reads a lane's last row (a lane reading more is refused by name,
        // `serve::readout_program`).
        let share = match self.readout_store_rows() {
            Some(store) => (store / lanes.max(1)).max(1),
            None => 1,
        };
        // Every lane at the page bound: its slot's every page, its rows
        // last.
        let owned: Vec<(Vec<u32>, Vec<u32>, crate::store::Scratch, Option<Masking>)> = rows
            .iter()
            .map(|&n| {
                let n = n.min(context);
                (
                    vec![0; n as usize],
                    (n.saturating_sub(share)..n).collect(),
                    self.pools.scratch(0, context.div_ceil(page) as usize),
                    with_mask.then(|| {
                        Masking::Extent(Mask {
                            runs: vec![0, context],
                            total: u64::from(context),
                        })
                    }),
                )
            })
            .collect();
        let replays = self.pools.rs_shape().is_some();
        // The rung's carried room, spread over the lanes.
        let room = crate::rs::carried_room(rung);
        let carried_of = |at: u32| room / lanes + u32::from(at < room % lanes);
        let verbs: Vec<RsVerb> = owned
            .iter()
            .enumerate()
            .map(|(at, (tokens, _, scratch, _))| {
                let first = context - tokens.len() as u32;
                let carried = carried_of(at as u32);
                if replays {
                    RsVerb::Buffer {
                        pages: scratch.rs.clone(),
                        base: 0,
                        folded: first.saturating_sub(carried),
                        fold: FoldLen::Host(carried),
                        lineage: engine::fire::Lineage::Set,
                    }
                } else {
                    RsVerb::Fold
                }
            })
            .collect();
        // Every declared guest input fed on every lane, a cell of its whole
        // size.
        let seats = self.dit.input_seats();
        let cells: Vec<Vec<(u32, Vec<u8>)>> = rows
            .iter()
            .map(|&n| {
                seats
                    .iter()
                    .map(|&(id, per_lane, row)| {
                        let count = if per_lane { 1 } else { n.min(context) as usize };
                        (id, vec![0u8; count * row])
                    })
                    .collect()
            })
            .collect();
        let feeds: Vec<Vec<crate::dit::InputCell<'_>>> = cells
            .iter()
            .map(|lane| {
                lane.iter()
                    .map(|(id, bytes)| crate::dit::InputCell { id: *id, bytes })
                    .collect()
            })
            .collect();
        for slot in 0..lanes.min(slots) {
            self.open(slot)?;
        }
        self.token_feeds = rows
            .iter()
            .enumerate()
            .map(|(lane, &n)| TokenFeed {
                lane,
                at: vec![(0, 0); n.min(context) as usize],
            })
            .collect();
        let seated: Vec<Seated<'_>> = owned
            .iter()
            .enumerate()
            .zip(verbs.iter().zip(&feeds))
            .map(|((at, (tokens, readout, scratch, mask)), (verb, inputs))| {
                let mut seated = Seated::of(Lane {
                    slot: at as u32 % slots,
                    word,
                    tokens,
                });
                seated.pages = &scratch.pages;
                seated.window = &scratch.window;
                seated.fixed = &scratch.fixed;
                seated.held = Some(context - tokens.len() as u32);
                seated.readout = Some(readout);
                seated.mask = mask.as_ref();
                seated.rs = verb;
                seated.inputs = inputs;
                seated
            })
            .collect();
        self.sizing = Some(0);
        self.fire_full(&seated)?;
        Ok(self.sizing.unwrap_or(0))
    }
}
