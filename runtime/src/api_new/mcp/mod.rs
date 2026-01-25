//! pie:mcp - MCP (Model Context Protocol) client interface

pub mod types;
pub mod client;

pub use types::{FutureContent, FutureJsonString};
pub use client::Session;
