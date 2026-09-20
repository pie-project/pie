#![allow(unsafe_code)]
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod boot;
mod engine;
mod log;

#[cfg(target_arch = "wasm32")]
mod page;

pub use boot::{BootConfig, BootSummary, Mount, boot};
