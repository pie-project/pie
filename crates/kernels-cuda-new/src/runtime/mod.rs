pub mod error;
pub mod nvrtc;
pub mod stream;

pub use error::Error;

/// Why a geometry could not be evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ungeometric {
    /// An axis is zero, so the grid covers nothing.
    Empty,
}

impl core::fmt::Display for Ungeometric {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => f.write_str("an axis is zero"),
        }
    }
}

/// One launch's geometry, as `KernelModule::fire` takes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per axis.
    pub grid: [u32; 3],
    /// Threads per block per axis.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
}
pub use stream::Stream;
