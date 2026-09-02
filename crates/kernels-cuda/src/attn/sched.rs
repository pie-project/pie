//! The schedulers' shared surface: the aligned bump allocator that carves
//! offsets out of a granted workspace, the staging buffer whose bytes become
//! a plan's `int_upload`, the host-table validators every planner opens
//! with, and the cost heap the load balancers ride.
//!
//! The planners themselves are native reimplementations of FlashInfer's host
//! scheduling: valid and deterministic, but not byte-identical to the C++
//! reference. Only the staged encoding (i32 little-endian vectors at
//! 16-byte-aligned offsets) is part of the contract.

use core::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::error::Error;

use crate::jit::refuse;

/// An aligned bump allocator that carves offsets out of a granted
/// workspace. It carries the plan op's name so an overflow refuses with
/// attribution instead of a bare capacity number.
#[derive(Clone, Copy, Debug)]
pub struct AlignedAllocator {
    op: &'static str,
    allocated: usize,
    remaining: usize,
}

impl AlignedAllocator {
    #[must_use]
    pub const fn new(op: &'static str, space: usize) -> Self {
        Self {
            op,
            allocated: 0,
            remaining: space,
        }
    }

    /// The offset lands as the `u32` the info tables carry — a schedule
    /// table past 4 GiB is refused here, once, not at every site.
    pub fn alloc(
        &mut self,
        size: usize,
        alignment: usize,
        what: &'static str,
    ) -> Result<u32, Error> {
        let padding = if alignment > 1 {
            (alignment - (self.allocated % alignment)) % alignment
        } else {
            0
        };
        if padding > self.remaining || size > self.remaining - padding {
            return Err(refuse(
                self.op,
                format!(
                    "`{what}` does not fit the granted workspace: {size} bytes asked, \
                     {} left",
                    self.remaining
                ),
            ));
        }
        let result = self.allocated + padding;
        self.allocated = result + size;
        self.remaining -= padding + size;
        u32::try_from(result)
            .map_err(|_| refuse(self.op, format!("`{what}` lands past any u32 offset")))
    }

    #[must_use]
    pub const fn used(&self) -> usize {
        self.allocated
    }
}

/// The host image of the int workspace, written at the offsets a layout
/// pass assigned and handed off as the plan's `int_upload`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staging {
    op: &'static str,
    bytes: Vec<u8>,
}

impl Staging {
    #[must_use]
    pub fn new(op: &'static str, len: usize) -> Self {
        Self {
            op,
            bytes: vec![0u8; len],
        }
    }

    pub fn put_i32s(
        &mut self,
        offset: usize,
        values: &[i32],
        what: &'static str,
    ) -> Result<(), Error> {
        let len = values.len() * 4;
        self.check(offset + len, len, what)?;
        for (slot, value) in self.bytes[offset..offset + len]
            .chunks_exact_mut(4)
            .zip(values)
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    pub fn put_i32(&mut self, offset: usize, value: i32, what: &'static str) -> Result<(), Error> {
        self.put_i32s(offset, &[value], what)
    }

    pub fn put_bools(
        &mut self,
        offset: usize,
        values: impl Iterator<Item = bool>,
        what: &'static str,
    ) -> Result<(), Error> {
        for (i, v) in values.enumerate() {
            self.check(offset + i + 1, 1, what)?;
            self.bytes[offset + i] = u8::from(v);
        }
        Ok(())
    }

    #[must_use]
    pub fn into_upload(mut self, len: usize) -> Vec<u8> {
        self.bytes.truncate(len);
        self.bytes
    }

    fn check(&self, end: usize, size: usize, what: &'static str) -> Result<(), Error> {
        if end > self.bytes.len() {
            return Err(refuse(
                self.op,
                format!(
                    "`{what}` writes past the staged workspace: {size} bytes at the tail, \
                     {} staged",
                    self.bytes.len()
                ),
            ));
        }
        Ok(())
    }
}

/// Walks a host indptr once and hands back the per-request span widths it
/// spells. Indptrs are engine-bound host twins, so a short or non-monotone
/// table is refused, not asserted.
pub fn spans(
    op: &'static str,
    which: &'static str,
    indptr: &[i32],
    batch: usize,
) -> Result<Vec<u32>, Error> {
    if indptr.len() < batch + 1 {
        return Err(refuse(
            op,
            format!(
                "the host {which} holds {} entries for a batch of {batch}",
                indptr.len()
            ),
        ));
    }
    let mut widths = Vec::with_capacity(batch);
    for pair in indptr[..=batch].windows(2) {
        let width = i64::from(pair[1]) - i64::from(pair[0]);
        if width < 0 {
            return Err(refuse(op, format!("the host {which} is not monotone")));
        }
        // Two i32 endpoints bound the difference under 2^32.
        widths.push(width as u32);
    }
    Ok(widths)
}

/// A per-request length table: one non-negative entry per request.
pub fn lengths(
    op: &'static str,
    which: &'static str,
    table: &[i32],
    batch: usize,
) -> Result<Vec<u32>, Error> {
    if table.len() < batch {
        return Err(refuse(
            op,
            format!(
                "the host {which} holds {} entries for a batch of {batch}",
                table.len()
            ),
        ));
    }
    table[..batch]
        .iter()
        .map(|&len| {
            u32::try_from(len)
                .map_err(|_| refuse(op, format!("the host {which} holds a negative length")))
        })
        .collect()
}

/// A laid-out offset at its stage write. The layout pass in the same
/// builder assigned it lines above, so `None` is a builder bug — an
/// invariant, not an input.
#[must_use]
pub fn at(offset: Option<u32>) -> usize {
    offset.expect("the layout pass assigned this offset") as usize
}

/// Narrows a host-computed schedule value to the i32 the device text reads.
pub fn narrow(op: &'static str, what: &'static str, value: i64) -> Result<i32, Error> {
    i32::try_from(value)
        .map_err(|_| refuse(op, format!("`{what}` reaches {value}, past the device's i32")))
}

/// [`narrow`], over a whole staged vector.
pub fn narrow_all(op: &'static str, what: &'static str, values: &[i64]) -> Result<Vec<i32>, Error> {
    values.iter().map(|&v| narrow(op, what, v)).collect()
}

#[derive(Clone, Copy, Debug)]
struct Lane {
    cost: f32,
    id: u32,
}

impl PartialEq for Lane {
    fn eq(&self, other: &Self) -> bool {
        matches!(self.cmp(other), Ordering::Equal)
    }
}

impl Eq for Lane {}

impl PartialOrd for Lane {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Lane {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// The load balancer's min-heap over `(accumulated cost, lane)`: `pop`
/// hands back the least-loaded lane. Ties resolve to the lower lane id, so
/// a fixed input always balances the same way.
#[derive(Clone, Debug)]
pub struct CostHeap {
    heap: BinaryHeap<Reverse<Lane>>,
}

impl CostHeap {
    /// One lane per CTA (or cluster), all starting unloaded.
    #[must_use]
    pub fn new(lanes: u32) -> Self {
        Self {
            heap: (0..lanes).map(|id| Reverse(Lane { cost: 0.0, id })).collect(),
        }
    }

    /// The least-loaded lane and its accumulated cost.
    pub fn pop(&mut self) -> (u32, f32) {
        let Reverse(lane) = self.heap.pop().expect("CostHeap::pop on an empty heap");
        (lane.id, lane.cost)
    }

    pub fn insert(&mut self, id: u32, cost: f32) {
        self.heap.push(Reverse(Lane { cost, id }));
    }
}

/// The load balancer's cost of one work item.
#[must_use]
pub fn cost_function(qo_len: u32, kv_len: u64) -> f32 {
    2.0 * (qo_len as f32) + (kv_len as f32)
}

/// Where a causal tile's kv walk ends, in packed coordinates.
#[must_use]
pub fn packed_causal_kv_end(
    qo_len: u32,
    kv_len: u32,
    qo_tile_idx: u32,
    cluster_tile_q: u32,
    num_qo_tiles: u32,
    group_size: u32,
) -> u32 {
    if qo_tile_idx + 1 == num_qo_tiles {
        return kv_len;
    }
    let init = i64::from(kv_len) - i64::from(qo_len);
    let walked = (i64::from(qo_tile_idx) + 1) * i64::from(cluster_tile_q);
    let end = init + (walked + i64::from(group_size) - 1) / i64::from(group_size);
    // Clamped into [0, kv_len], so the u32 round trip is exact.
    end.clamp(0, i64::from(kv_len)) as u32
}
