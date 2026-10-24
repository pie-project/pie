//! Expert-parallel wiring of one process's ranks: each rank's EP signal
//! block, landing (the rows its peers dispatch to it) and return (the
//! outputs its peers combine back to it), mapped into every rank. Peers
//! write only into these fixed buffers: each rank's arena is laid out per
//! program and bucket, so no peer knows where a rank's IR values live. The
//! values a dispatch or combine makes are seated on them instead of the
//! arena ([`Seats`]), so the rows land where their readers read. Nothing
//! here is taken while the ranks fire: an allocation between two ranks'
//! launches can wait on a peer spinning for the rank that waits.
#![allow(unsafe_code)]

use kernels_cuda::Tensor;
use kernels_cuda::collective::{EP_MAX_RANKS, EP_SIGNAL_BYTES, Ep, Landing, ep_scratch_bytes};

use crate::error::{Fault, Result};
use crate::run::SlotTable;
use poem_compiler::ArenaMap;
use poem_compiler::arena::Placement as Slot;
use poem_ir::{Collective, Dtype, Operands, Operation, Trace, ValueId};

/// The widest step any dispatch or combine of the group fires: sizes the
/// fixed buffers once, at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Widest {
    /// The largest token bucket a rank fires.
    pub tokens: u32,
    /// Bytes of a dispatched token row.
    pub row_bytes: u64,
    /// Bytes of an expert output row.
    pub out_bytes: u64,
    /// The widest fan-out.
    pub top_k: u32,
}

/// What a rank's trace dispatches: every `(layer, experts, slots)`, and the
/// widest step a dispatch or combine of it fires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub layers: Vec<(u32, u32, u32)>,
    pub widest: Widest,
}

/// One rank's part of the wiring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wire {
    pub ep: Ep,
    widest: Widest,
    rows: u64,
    local: u64,
    row_weights: u64,
    origin: u64,
    routed: u64,
}

impl Wire {
    /// This rank's landing at a step of `tokens` (every rank's bucket) and
    /// `k` routes a token, viewed as the kernels take it: `world * tokens`
    /// rows of `width` elements of `dtype`, dense from each buffer's start.
    pub fn landing(&self, tokens: u32, k: u32, width: u32, dtype: Dtype, weights: bool) -> Landing {
        let cap = self.ep.world * tokens;
        Landing {
            rows: Tensor::new(self.rows, cap, width, dtype),
            local: Tensor::new(self.local, cap, k, Dtype::I32),
            row_weights: weights.then(|| Tensor::new(self.row_weights, cap, k, Dtype::F32)),
            origin: Tensor::new(self.origin, cap, 1, Dtype::I32),
        }
    }

    /// This rank's return at a step of `tokens` and `k` routes a token:
    /// `tokens * k` rows of `width` elements of `dtype`.
    pub fn routed(&self, tokens: u32, k: u32, width: u32, dtype: Dtype) -> Tensor {
        Tensor::new(self.routed, tokens * k, width, dtype)
    }

    /// Whether a step of `tokens` rows `row_bytes` wide (outputs `out_bytes`)
    /// fits the buffers sized at load.
    #[must_use]
    pub fn holds(&self, tokens: u32, row_bytes: u64, out_bytes: u64, top_k: u32) -> bool {
        let w = self.widest;
        tokens <= w.tokens
            && row_bytes <= w.row_bytes
            && out_bytes <= w.out_bytes
            && top_k == w.top_k
    }
}

/// Which of a rank's fixed buffers a dispatch or combine output sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seat {
    Rows,
    Local { k: u32 },
    RowWeights { k: u32 },
    Origin,
    Routed { k: u32 },
}

/// The values of one arena map seated on the rank's fixed buffers, by id:
/// every dispatch and combine output and whatever the map aliases to one
/// (an op writing in place into the rows it read).
///
/// A peer writes into a rank's landing once every rank has entered the next
/// dispatch, and into its return once every rank has sent that dispatch's
/// rows: so a landing's values are read before the next dispatch, and a
/// return's no later than it. [`Seats::of`] refuses a trace that reads one
/// past that.
#[derive(Clone, Debug, Default)]
pub struct Seats(Vec<Option<Seat>>);

impl Seats {
    pub fn of(trace: &Trace, map: &ArenaMap) -> std::result::Result<Seats, String> {
        let root = |mut value: ValueId| {
            for _ in 0..=map.placements.len() {
                match map.placements.get(value.0 as usize) {
                    Some(Slot::Alias(to)) => value = *to,
                    _ => break,
                }
            }
            value
        };
        let mut made: Vec<Option<(Seat, usize)>> = vec![None; trace.values.len()];
        for (at, node) in trace.nodes.iter().enumerate() {
            let mut seat = |value: ValueId, seat| made[value.0 as usize] = Some((seat, at));
            match &node.op {
                Operation::Collective(Collective::Dispatch {
                    top_k,
                    rows,
                    local,
                    row_weights,
                    origin,
                    ..
                }) => {
                    seat(*rows, Seat::Rows);
                    seat(*local, Seat::Local { k: *top_k });
                    if let Some(w) = row_weights {
                        seat(*w, Seat::RowWeights { k: *top_k });
                    }
                    seat(*origin, Seat::Origin);
                }
                Operation::Collective(Collective::Combine { top_k, routed, .. }) => {
                    seat(*routed, Seat::Routed { k: *top_k });
                }
                _ => {}
            }
        }
        let seated: Vec<Option<(Seat, usize)>> = (0..trace.values.len())
            .map(|v| made[root(ValueId(v as u32)).0 as usize])
            .collect();
        let dispatches: Vec<usize> = (trace.nodes.iter().enumerate())
            .filter(|(_, n)| matches!(n.op, Operation::Collective(Collective::Dispatch { .. })))
            .map(|(at, _)| at)
            .collect();
        let mut ins = Vec::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            ins.clear();
            node.op.inputs(&mut ins);
            for v in &ins {
                let Some((seat, from)) = seated[v.0 as usize] else {
                    continue;
                };
                let Some(&next) = dispatches.iter().find(|&&d| d > from) else {
                    continue;
                };
                let past = match seat {
                    Seat::Routed { .. } => at > next,
                    _ => at >= next,
                };
                if past {
                    return Err(format!(
                        "node {at} reads value {}, which node {from} seated on this rank's \
                         expert-parallel buffers and the dispatch at node {next} hands back \
                         to its peers",
                        v.0
                    ));
                }
            }
        }
        Ok(Seats(
            seated
                .into_iter()
                .map(|s| s.map(|(seat, _)| seat))
                .collect(),
        ))
    }

    /// Points each seated slot of `slots` at `wire`'s buffers for a step at
    /// token `bucket`. A slot the wire cannot hold keeps its arena rows, and
    /// the arm that writes it refuses.
    pub fn seat(&self, wire: &Wire, bucket: u32, slots: &mut SlotTable) {
        for (cell, seat) in slots.0.iter_mut().zip(&self.0) {
            let (Some(slot), Some(seat)) = (cell.as_mut(), seat) else {
                continue;
            };
            let row_bytes =
                u64::from(slot.width) * poem_compiler::arena::elem_bytes(slot.dtype).unwrap_or(0);
            let fits = match seat {
                Seat::Rows => wire.holds(bucket, row_bytes, 0, wire.widest.top_k),
                Seat::Routed { .. } => wire.holds(bucket, 0, row_bytes, wire.widest.top_k),
                _ => wire.holds(bucket, 0, 0, wire.widest.top_k),
            };
            if !fits {
                continue;
            }
            let landing = |k| wire.landing(bucket, k, slot.width, slot.dtype, true);
            *slot = match *seat {
                Seat::Rows => landing(1).rows,
                Seat::Local { k } => landing(k).local,
                Seat::RowWeights { k } => landing(k).row_weights.unwrap_or(*slot),
                Seat::Origin => landing(1).origin,
                Seat::Routed { k } => wire.routed(bucket, k, slot.width, slot.dtype),
            };
        }
    }
}

/// Wires `ordinals` (one rank each, in rank order) for expert parallelism
/// at `widest`. The ranks must reach one another as the peer collectives
/// do ([`crate::comm::open_peers`] checks the path first); there is no
/// fallback for a dispatch.
pub fn wire(ordinals: &[i32], widest: Widest) -> Result<Vec<Wire>> {
    let off = |why: String| Fault::program("ep::wire", why);
    let world = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    if !(2..=EP_MAX_RANKS).contains(&world) {
        return Err(off(format!(
            "expert parallelism runs 2 to 8 ranks, not {world}"
        )));
    }
    if crate::comm::open_peers(ordinals).is_none() {
        return Err(off(
            "expert parallelism needs every rank to map every other's memory (NVLink peers)".into(),
        ));
    }
    let cap = u64::from(world) * u64::from(widest.tokens);
    let k = u64::from(widest.top_k);
    let parts = [
        EP_SIGNAL_BYTES,
        cap * widest.row_bytes,
        cap * k * 4,
        cap * k * 4,
        cap * 4,
        u64::from(widest.tokens) * k * widest.out_bytes,
        ep_scratch_bytes(widest.tokens, world),
    ];
    let mut bases = Vec::with_capacity(ordinals.len());
    for &a in ordinals {
        bases.push(carve(a, ordinals, &parts)?);
        crate::device::ctx::bind_thread(a)?;
        kernels_cuda::collective::load_ep().map_err(crate::error::kernel)?;
    }
    let mut signals = [0u64; EP_MAX_RANKS as usize];
    for (s, b) in signals.iter_mut().zip(&bases) {
        *s = b[0];
    }
    Ok(bases
        .iter()
        .enumerate()
        .map(|(rank, b)| Wire {
            ep: Ep {
                rank: rank as u32,
                world,
                signals,
                scratch: b[6],
                scratch_tokens: widest.tokens,
            },
            widest,
            rows: b[1],
            local: b[2],
            row_weights: b[3],
            origin: b[4],
            routed: b[5],
        })
        .collect())
}

/// A rank's mapping: the signal block, the landing's rows, local slots,
/// weights and origins, the return, and the dispatch scratch.
const PARTS: usize = 7;

/// One mapping on `device` that `readers` reach, cut into `parts` 256-byte
/// aligned, the first (the signal block) zeroed.
#[cfg(feature = "cuda")]
fn carve(device: i32, readers: &[i32], parts: &[u64; PARTS]) -> Result<[u64; PARTS]> {
    use crate::device::ctx::check;
    use crate::device::rt;

    let at = |p: &[u64]| p.iter().map(|b| b.next_multiple_of(256)).sum::<u64>();
    let base = crate::device::elastic::shared(device, readers, at(parts))?;
    let mut cut = [0u64; PARTS];
    for (i, c) in cut.iter_mut().enumerate() {
        *c = base + at(&parts[..i]);
    }
    // SAFETY: the signal block is the first `EP_SIGNAL_BYTES` of the mapping
    // just made on `device`.
    unsafe {
        check("cudaSetDevice", rt::cudaSetDevice(device))?;
        check(
            "cudaMemset",
            rt::cudaMemset(
                cut[0] as *mut core::ffi::c_void,
                0,
                EP_SIGNAL_BYTES as usize,
            ),
        )?;
    }
    Ok(cut)
}

#[cfg(not(feature = "cuda"))]
fn carve(_: i32, _: &[i32], _: &[u64; PARTS]) -> Result<[u64; PARTS]> {
    Err(Fault::Runtimeless)
}

pub mod rebalance;

/// One MoE layer's placement on a rank's device: the table its dispatches
/// read and the loads they add each local slot's rows to.
struct Layer {
    place: kernels_cuda::collective::Placement,
    table: crate::device::Buffer,
    loads: crate::device::Buffer,
}

/// One plane of a routed bank on a rank: its `slots` slots, slot `s` the
/// `slot_bytes` at `ptr + s * slot_bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    pub ptr: u64,
    pub slot_bytes: u64,
}

/// A layer's routed banks on a rank, every plane in the trace's order, or
/// why its slots cannot move.
pub type Banks = std::collections::BTreeMap<u32, std::result::Result<Vec<Plane>, String>>;

/// A rank's expert-parallel state: every rank's wire (its own at `rank`)
/// and each lowered MoE layer's placement table on this device, by the
/// layer number the trace's dispatches carry. Several dispatches may share
/// a layer (a drafter block run at every draft step reads the same bank).
pub struct EpRank {
    pub wires: Vec<Wire>,
    pub rank: u32,
    layers: std::collections::BTreeMap<u32, Layer>,
    banks: Banks,
    seats: Seats,
    seats_forked: Seats,
}

impl EpRank {
    /// The uniform placement of each `(layer, experts, slots)` the trace
    /// dispatches, its table sized for any later rebalance.
    pub fn new(wires: Vec<Wire>, rank: u32, layers: &[(u32, u32, u32)]) -> Result<EpRank> {
        let ranks = wires.len() as u32;
        let mut held = std::collections::BTreeMap::new();
        for &(layer, experts, slots) in layers {
            if held.contains_key(&layer) {
                continue;
            }
            let placement = engine::placement::Placement::uniform(experts, ranks, slots)
                .map_err(|why| Fault::program("ep::EpRank", format!("layer {layer}: {why:?}")))?;
            let stride = engine::placement::Placement::stride_for(ranks * slots - experts + 1);
            let bytes: Vec<u8> = (placement.table(stride).iter())
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let mut table = crate::device::Buffer::zeroed(bytes.len())?;
            table.write(0, &bytes)?;
            let loads = crate::device::Buffer::zeroed(slots as usize * 4)?;
            let place = kernels_cuda::collective::Placement {
                table: table.ptr(),
                stride,
                experts,
                slots,
                loads: loads.ptr(),
            };
            held.insert(
                layer,
                Layer {
                    place,
                    table,
                    loads,
                },
            );
        }
        Ok(EpRank {
            wires,
            rank,
            layers: held,
            banks: Banks::new(),
            seats: Seats::default(),
            seats_forked: Seats::default(),
        })
    }

    /// The routed banks this rank holds, by layer: what a rebalance copies.
    pub fn hold_banks(&mut self, banks: Banks) {
        self.banks = banks;
    }

    /// A layer's bank planes on this rank, or why they cannot move.
    pub fn planes(&self, layer: u32) -> std::result::Result<&[Plane], String> {
        match self.banks.get(&layer) {
            Some(Ok(planes)) => Ok(planes),
            Some(Err(why)) => Err(why.clone()),
            None => Err(format!("layer {layer} holds no routed bank on this rank")),
        }
    }

    /// Each layer's `(layer, experts, slots, stride)`.
    pub fn layers(&self) -> impl Iterator<Item = (u32, u32, u32, u32)> + '_ {
        (self.layers.iter()).map(|(&l, layer)| {
            (
                l,
                layer.place.experts,
                layer.place.slots,
                layer.place.stride,
            )
        })
    }

    /// Each layer's rows per local slot since the last read, which it
    /// zeroes. The rank's dispatches must have finished.
    pub fn take_loads(&mut self) -> Result<std::collections::BTreeMap<u32, Vec<u32>>> {
        let mut out = std::collections::BTreeMap::new();
        for (&l, layer) in &mut self.layers {
            let mut bytes = vec![0u8; layer.loads.bytes()];
            layer.loads.read(0, &mut bytes)?;
            layer.loads.zero_span(0, bytes.len())?;
            let loads = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            out.insert(l, loads);
        }
        Ok(out)
    }

    /// Rewrites `layer`'s table in place: a captured body reads it where it
    /// was.
    pub fn write_table(&mut self, layer: u32, table: &[i32]) -> Result<()> {
        let layer = (self.layers.get_mut(&layer))
            .ok_or_else(|| Fault::program("ep::write_table", format!("no layer {layer}")))?;
        let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
        if bytes.len() != layer.table.bytes() {
            return Err(Fault::program(
                "ep::write_table",
                "a table of another stride than the one laid at load",
            ));
        }
        layer.table.write(0, &bytes)
    }

    /// Seats the values of `trace` on this rank's buffers, under the arena
    /// map in order and the one forked for side streams.
    pub fn seat(
        &mut self,
        trace: &Trace,
        map: &ArenaMap,
        forked: Option<&ArenaMap>,
    ) -> std::result::Result<(), String> {
        self.seats = Seats::of(trace, map)?;
        self.seats_forked = forked.map_or(Ok(Seats::default()), |m| Seats::of(trace, m))?;
        Ok(())
    }

    /// The seats of a fire that carves the forked map, or the one in order.
    #[must_use]
    pub fn seats(&self, forked: bool) -> &Seats {
        if forked {
            &self.seats_forked
        } else {
            &self.seats
        }
    }

    #[must_use]
    pub fn wire(&self) -> &Wire {
        &self.wires[self.rank as usize]
    }

    #[must_use]
    pub fn placement(&self, layer: u32) -> Option<kernels_cuda::collective::Placement> {
        self.layers.get(&layer).map(|l| l.place)
    }
}

#[cfg(test)]
mod tests {
    use poem_ir::{Dtype, Platform};

    use super::*;

    fn lowered() -> Trace {
        let whole = poem_compiler::catalog::trace_of(
            "qwen3-a3b-micro",
            Dtype::Bf16,
            Dtype::Bf16,
            Platform::Cuda,
        );
        poem_compiler::expert::lower(whole, 2, 0).expect("the micro's MoE lowers")
    }

    fn unaliased() -> ArenaMap {
        ArenaMap {
            placements: Vec::new(),
            spans: Vec::new(),
            live_in: Vec::new(),
            bytes: 0,
        }
    }

    #[test]
    fn every_dispatch_and_combine_output_is_seated() {
        let trace = lowered();
        let seats = Seats::of(&trace, &unaliased()).expect("the micro reads its landings in time");
        let mut outs = Vec::new();
        for node in &trace.nodes {
            if let Operation::Collective(Collective::Dispatch { .. } | Collective::Combine { .. }) =
                &node.op
            {
                outs.clear();
                node.op.outputs(&mut outs);
                let counts = match &node.op {
                    Operation::Collective(Collective::Dispatch { counts, .. }) => Some(*counts),
                    _ => None,
                };
                for v in outs.iter().filter(|&&v| Some(v) != counts) {
                    assert!(
                        seats.0[v.0 as usize].is_some(),
                        "value {} is not seated",
                        v.0
                    );
                }
            }
        }
    }

    #[test]
    fn a_landing_read_past_the_next_dispatch_is_refused() {
        let mut trace = lowered();
        let first = trace
            .nodes
            .iter()
            .position(|n| matches!(n.op, Operation::Collective(Collective::Dispatch { .. })))
            .expect("the micro dispatches");
        let Operation::Collective(Collective::Dispatch { rows, .. }) = trace.nodes[first].op else {
            unreachable!()
        };
        let mut ins = Vec::new();
        let reader = trace.nodes[first + 1..]
            .iter()
            .find(|n| {
                ins.clear();
                n.op.inputs(&mut ins);
                ins.contains(&rows)
            })
            .expect("the rows are read")
            .clone();
        assert!(
            trace
                .nodes
                .iter()
                .skip(first + 1)
                .any(|n| { matches!(n.op, Operation::Collective(Collective::Dispatch { .. })) }),
            "the micro dispatches more than once"
        );
        trace.nodes.push(reader);
        let why = Seats::of(&trace, &unaliased()).expect_err("a stale landing read");
        assert!(why.contains("hands back to its peers"), "{why}");
    }
}
