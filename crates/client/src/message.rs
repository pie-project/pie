//! Client ↔ server message vocabulary.
//!
//! The definitions now live on the dependency floor in
//! [`client_api::message`] so the floor-resident `edge` session frames can
//! embed them without the floor depending on this (tokio/websocket/crypto)
//! crate. Re-exported here so the historical `::client::message::*` path
//! keeps resolving for client consumers.

pub use client_api::message::*;
