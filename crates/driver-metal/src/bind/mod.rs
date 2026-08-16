//! Where a lowered launch meets the device: [`crate::lowering`] decides symbol,
//! grid and addresses host-side; compiling, [`tables`] and [`encode`] cannot be.

pub mod encode;
pub mod tables;
