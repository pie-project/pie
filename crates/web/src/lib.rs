//! pie in a browser tab.
//!
//! The page loads this module, hands it a model artifact and an inferlet, and
//! then talks the client protocol to it: the same `ClientMessage` /
//! `ServerMessage` a WebSocket carries natively, as JSON strings. Everything
//! asynchronous is a task on `web-rt`'s executor; the page drives that
//! executor by calling `pie_tick` (through `WebAssembly.promising`, since a
//! tick may switch fibers) whenever `pie_wake` asks it to.
//!
//! Exports, in the order a page uses them:
//! `pie_init` → `pie_boot` → `pie_install_program` → `pie_open_session` /
//! `pie_send` / `pie_recv` / `pie_close_session`, with `pie_tick` underneath.

#![allow(unsafe_code)]
// The page-facing half only exists on wasm32; natively the crate is checked
// and clippy'd, not used.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod boot;
mod engine;
mod log;

#[cfg(target_arch = "wasm32")]
mod page;

pub use boot::{BootConfig, BootSummary, Mount, boot};
