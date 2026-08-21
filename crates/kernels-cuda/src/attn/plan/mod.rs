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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {

    pub num_sm: u32,
    pub cc_major: i32,
}

impl Device {

    #[must_use]
    pub const fn new(num_sm: u32, cc_major: i32) -> Self {
        Self { num_sm, cc_major }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workspace {

    pub float_bytes: usize,
    pub int_bytes: usize,
}

impl Workspace {

    #[must_use]
    pub const fn new(float_bytes: usize, int_bytes: usize) -> Self {
        Self { float_bytes, int_bytes }
    }

    #[must_use]
    pub const fn unbounded() -> Self {
        Self { float_bytes: usize::MAX, int_bytes: usize::MAX }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan<I> {

    pub info: I,
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub float_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {

    pub float_bytes: usize,
    pub int_bytes: usize,
}
