//! Client ↔ server message vocabulary.
//!
//! The definitions now live on the dependency floor in
//! [`pie_schema::message`] so the floor-resident `edge` session frames can
//! embed them without the floor depending on this (tokio/websocket/crypto)
//! crate. Re-exported here so the historical `pie_client::message::*` path
//! keeps resolving for client consumers.

pub use pie_schema::message::*;
