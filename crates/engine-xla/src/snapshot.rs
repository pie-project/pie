//! Snapshots inside a pass (CONTRACT-BC 16): a lane's recurrent state as of
//! interior rows, each landed in a record of its own.
//!
//! The scans on this backend (the gated delta's chunked HLO and Mosaic
//! forms, the conv) land only the state after a lane's last row, so a
//! fire whose lanes snapshot is fired in pieces: every snapshotting lane's
//! rows up to each snapshot row fire alone, the record the lane writes is
//! copied whole into the snapshot's record after each piece, and the last
//! piece fires with the step's other lanes. Each piece is the fire a pass
//! cut at that row would have sent (its word re-classed for its rows, the
//! record in place after the first), so every snapshot, the lane's record
//! and its readout are what the cut pass lands, bit for bit. The cost is
//! the cut pass's: a fire per snapshot row, on the stream behind each
//! other, where a scan writing the state as it walks would take one.

use engine::fire::{Lane, Readout, RsReset, Step};

use crate::error::Fault;

/// The most snapshots one lane lands in a fire.
pub const MOST: u32 = 4;

/// The word bits a piece re-classes: the single-token fact and the fresh
/// fact.
#[derive(Debug, Clone, Copy)]
pub struct Bits {
    pub single: u64,
    pub fresh: u64,
}

impl Bits {
    #[must_use]
    pub fn of(facts: &poem_ir::Facts) -> Bits {
        use poem_ir::Request;
        let plain = facts.word(&Request::new(2, false));
        Bits {
            single: plain ^ facts.word(&Request::new(1, false)),
            fresh: plain ^ facts.word(&Request::new(2, false).starting(true)),
        }
    }
}

/// One piece of a snapshotting step: the step to fire, and after it the
/// whole-slot copies `(shape, src, dst)` that land its snapshots.
#[derive(Debug)]
pub struct Piece {
    pub step: Step,
    pub copies: Vec<(engine::chunk::ShapeId, u32, u32)>,
}

/// `step` cut at its lanes' snapshot rows: the pieces before the last, each
/// one lane alone, then the step itself with every snapshotting lane cut
/// to its rows past its last snapshot. `records` are the load's fixed
/// shapes, in order: a record is a slot of each a piece.
pub fn pieces(
    step: &Step,
    bits: Bits,
    records: &[engine::chunk::ShapeId],
) -> Result<Vec<Piece>, Fault> {
    let refuse = |lane: usize, why: &str| Fault::Program {
        at: "snapshot",
        why: format!("lane {lane} snapshots its recurrent state and {why}"),
    };
    let mut out = Vec::new();
    let mut last = step.clone();
    for (at, lane) in step.lanes.iter().enumerate() {
        if lane.rs_snapshots.is_empty() {
            continue;
        }
        if step.attachments.iter().any(|a| a.lane as usize == at)
            || step.media.iter().any(|m| m.lane as usize == at)
        {
            return Err(refuse(at, "is bound to a program or carries media"));
        }
        if lane.mask.is_some()
            || !lane.inputs.is_empty()
            || !lane.channels.is_empty()
            || lane.tap.is_some()
            || matches!(lane.readout, Readout::Rows(_))
        {
            return Err(refuse(
                at,
                "carries a mask, inputs, channels, a tap or reads rows out: a piece cannot \
                 carry them",
            ));
        }
        // The lane's own record: each fixed shape's pieces it writes.
        let mut own = Vec::with_capacity(records.len());
        for &shape in records {
            let Some((_, run_cow)) = lane.fixed.iter().find(|(id, _)| *id == shape) else {
                return Err(refuse(
                    at,
                    &format!("names no record of fixed shape {shape}"),
                ));
            };
            own.push((
                shape,
                run_cow.iter().map(|&[_, cow]| cow).collect::<Vec<u32>>(),
            ));
        }
        let mut from = 0u32;
        for snap in &lane.rs_snapshots {
            let mut piece = cut(lane, from, snap.row, bits);
            piece.readout = Readout::None;
            // The snapshot's record: every fixed shape's pieces where the
            // runtime names them (`RsSnapshot::fixed`), else the first
            // shape's alone, the load's only one.
            let named: Vec<(engine::chunk::ShapeId, Vec<u32>)> = if snap.fixed.is_empty() {
                if records.len() != 1 {
                    return Err(refuse(
                        at,
                        &format!(
                            "names its snapshot's record in the first of {} fixed shapes only",
                            records.len()
                        ),
                    ));
                }
                let pieces = if snap.pieces.is_empty() {
                    vec![snap.slot]
                } else {
                    snap.pieces.clone()
                };
                vec![(records[0], pieces)]
            } else {
                snap.fixed.clone()
            };
            let mut copies = Vec::new();
            for (shape, cows) in &own {
                let Some((_, dst)) = named.iter().find(|(id, _)| id == shape) else {
                    return Err(refuse(
                        at,
                        &format!("names no snapshot record of fixed shape {shape}"),
                    ));
                };
                if dst.len() != cows.len() {
                    return Err(refuse(
                        at,
                        &format!(
                            "names a snapshot record of {} pieces of shape {shape} where its \
                             own has {}",
                            dst.len(),
                            cows.len()
                        ),
                    ));
                }
                copies.extend(cows.iter().zip(dst).map(|(&src, &dst)| (*shape, src, dst)));
            }
            out.push(Piece {
                step: Step {
                    lanes: vec![piece],
                    attachments: Vec::new(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                },
                copies,
            });
            from = snap.row;
        }
        last.lanes[at] = cut(lane, from, lane.tokens.len() as u32, bits);
    }
    out.push(Piece {
        step: last,
        copies: Vec::new(),
    });
    Ok(out)
}

/// Rows `[from, to)` of `lane`, as a pass cut there would fire them.
fn cut(lane: &Lane, from: u32, to: u32, bits: Bits) -> Lane {
    let (a, b) = (from as usize, to as usize);
    let rows = to - from;
    let mut piece = lane.clone();
    piece.tokens = lane.tokens[a..b].to_vec();
    if !lane.positions.is_empty() {
        piece.positions = lane.positions[a..b].to_vec();
    }
    piece.rs_snapshots = Vec::new();
    piece.word = if rows == 1 {
        lane.word | bits.single
    } else {
        lane.word & !bits.single
    };
    if from > 0 {
        piece.kv.held = lane.kv.held + from;
        piece.word &= !bits.fresh;
        if lane.rs_reset == RsReset::Fresh {
            piece.rs_reset = RsReset::Held;
        }
        // The record is the copy the first piece wrote: in place from here.
        for (_, pairs) in &mut piece.fixed {
            for pair in pairs.iter_mut() {
                *pair = [pair[1], pair[1]];
            }
        }
        // Slot copies run once, before the first piece reads or writes.
        piece.slot_copies = Vec::new();
    }
    piece
}
