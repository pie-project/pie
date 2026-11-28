//! The lane scratch table: where each lane of a grouped kernel finds its
//! value scratch. A grouped (batched) kernel runs a batch's lanes in one
//! dispatch, and lane `l`'s value scratch starts at entry `l` of this
//! table: `scratch = base(lane)`, every value at `scratch + offsets[v]`
//! as before. Nothing ties one lane's scratch to another's, so each
//! instance's scratch can lie in its own `Instance` slots.
//!
//! The ABI an engine binds (what this module states, eta owns it):
//! - name `lane_scratch` ([`LANE_SCRATCH`]), element `u64`
//!   ([`LaneScratch::ELEMENT_BYTES`]), length the dispatch's lanes (entry
//!   `l` for the batch's lane `l`, the kernels' `dispatch_lane`);
//! - bound at the SIMT kernels' parameter [`LaneScratch::simt_param`] and
//!   Metal's `[[buffer(n)]]` [`LaneScratch::metal_buffer`];
//! - an entry is the lane's scratch base in its backend's scratch address
//!   space ([`Backend::lane_base`]): a device address for CUDA / HIP and
//!   Metal, whose pools span several allocations (a member's scratch may
//!   lie in any of them, reached by address, resident by its engine).
//!
//! A lane's scratch is one piece: a grouped kernel addresses every value
//! of the lane from its one base, so the lane's stage scratch lies whole
//! in one contiguous run (an `eta_exec::lay_out` layout). An engine that would place a lane's scratch in
//! several pieces refuses that by name.
//!
//! The WGSL / SPIR-V steps and XLA's lowering run no grouped kernel: each
//! dispatch serves one instance and binds its scratch itself (WGSL's one
//! `heap` binding at the instance's offset; XLA's buffers), so they read no
//! table. Were they to batch lanes, their entry would be a byte offset into
//! their one bound pool buffer ([`LaneBase::Offset`]).

use crate::codegen::program::Backend;

/// What a lane scratch table's entry is in a backend's scratch address
/// space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneBase {
    /// The device address of the lane's first scratch byte.
    Address,
    /// The byte offset of the lane's first scratch byte into the one pool
    /// buffer the kernel binds.
    Offset,
}

/// The lane scratch table's binding ([`LANE_SCRATCH`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneScratch {
    /// Its name in every emitted kernel.
    pub name: &'static str,
    /// Its place among the SIMT grouped kernels' parameters, from 0.
    pub simt_param: usize,
    /// Its `[[buffer(n)]]` in Metal's grouped kernels.
    pub metal_buffer: usize,
}

impl LaneScratch {
    /// An entry's bytes: a `u64`, little-endian.
    pub const ELEMENT_BYTES: usize = 8;

    /// The table's bytes for `lanes` lanes.
    #[must_use]
    pub const fn bytes(lanes: u32) -> u64 {
        lanes as u64 * Self::ELEMENT_BYTES as u64
    }
}

pub const LANE_SCRATCH: LaneScratch = LaneScratch {
    name: "lane_scratch",
    simt_param: 6,
    metal_buffer: 4,
};

impl Backend {
    /// What an entry of [`LANE_SCRATCH`] is on this backend.
    #[must_use]
    pub const fn lane_base(self) -> LaneBase {
        match self {
            Backend::Cuda | Backend::Hip | Backend::Metal => LaneBase::Address,
        }
    }
}
