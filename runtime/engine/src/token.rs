//! Internal gateway↔engine trust token.
//!
//! Authentication of external clients is not the inference engine's job — the
//! trusted edge (gateway / reverse proxy) does that and forwards an identity
//! the engine trusts (see `server::Session`, which serves every session as
//! `internal`). What survives from the removed `auth` module is this one
//! anchor: a per-boot random secret the engine hands back through the
//! bootstrap handshake so an embedding host (the worker) can label the
//! trusted control channel. It is opaque and never verified inside the engine.
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;

/// Generate a fresh 48-byte URL-safe internal token for this engine boot.
pub fn generate_internal_token() -> String {
    let mut bytes = [0u8; 48];
    rand::rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}
