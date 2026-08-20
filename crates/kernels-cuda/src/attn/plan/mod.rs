pub mod alloc;
pub mod arith;
pub mod decode;
pub mod error;
pub mod heap;
pub mod info;
pub mod mla;
pub mod prefill;
pub mod sm90;
pub mod sort;

pub use error::Error;
pub use info::{DecodePlanInfo, MlaPlanInfo, PrefillPlanInfo, PrefillPlanSm90Info};

/// The device facts a planner reads, hoisted into a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    /// `cudaDevAttrMultiProcessorCount`.
    pub num_sm: u32,
    /// `cudaDevAttrComputeCapabilityMajor`.
    pub cc_major: i32,
}

impl Device {
    /// A device, named by the two attributes the planners read.
    #[must_use]
    pub const fn new(num_sm: u32, cc_major: i32) -> Self {
        Self { num_sm, cc_major }
    }
}

/// The two workspace buffers, as sizes — because a planner only ever needs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workspace {
    /// `float_workspace_size_in_bytes` — the partial-output arena, carved but
    pub float_bytes: usize,
    /// `int_workspace_size_in_bytes` — the descriptor arena, which the planner
    pub int_bytes: usize,
}

impl Workspace {
    /// A workspace of the two given sizes.
    #[must_use]
    pub const fn new(float_bytes: usize, int_bytes: usize) -> Self {
        Self { float_bytes, int_bytes }
    }

    /// The workspace a sizing pass uses: unbounded, so nothing refuses.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self { float_bytes: usize::MAX, int_bytes: usize::MAX }
    }
}

/// A finished plan: the params block the kernel reads, and the bytes to put under it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan<I> {
    /// The params block the device kernel reads. Its layout is the contract; see
    pub info: I,
    /// Exactly the bytes upstream would have copied H2D: the int workspace
    pub int_upload: Vec<u8>,
    /// `num_allocated_bytes()` of the int allocator: the length of
    pub int_bytes: usize,
    /// `num_allocated_bytes()` of the float allocator — carved, never written.
    pub float_bytes: usize,
}

/// What a sizing pass answers: how big the two arenas must be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    /// Bytes the float arena needs.
    pub float_bytes: usize,
    /// Bytes the int arena needs.
    pub int_bytes: usize,
}
