//! Host-side performance probes, organized by domain.
//!
//! Each subdomain (`fire`, eventually `chain_ext`, `inferlet`, etc.)
//! lives in its own module and is gated by its own Cargo feature
//! (`profile-fire`, etc.). Probe call sites use the domain-prefixed
//! macro (`probe_fire!`) so the *intent* of the probe is visible at
//! the call site without grepping.
//!
//! ## Why per-domain features
//!
//! Different probe domains have very different cost profiles:
//!
//! | Domain     | Rate              | When you'd enable                |
//! |------------|-------------------|----------------------------------|
//! | `fire`     | ~90×/sec @ conc=256 | scheduler optimisation         |
//! | `chain_ext`| ~23 000×/sec same | hunting fan-out wake latency     |
//! | `inferlet` | per program start | cold-start debugging             |
//! | `startup`  | one-shot          | boot-time tuning                 |
//!
//! Bundling them under one flag would make you pay chain-ext probe
//! cost (~0.3% throughput at conc=256) while investigating a one-shot
//! startup question. Per-domain flags let you opt into exactly the
//! scopes you need.
//!
//! ## Conventions
//!
//! - Probe-holder structs live in the domain module (e.g.
//!   `probe::fire::FireProbes`). They're always defined regardless of
//!   feature so callers and readers compile uniformly; only the
//!   `fetch_add` at probe sites is feature-gated.
//! - Macros are named `probe_<domain>!`. With the feature off they
//!   expand to a no-op that consumes their arguments without running
//!   `Instant::now()` or touching the atomic.
//! - Hierarchical probes (parent / child) are encoded as nested
//!   structs — `fire.execute.driver_fire_us`. The struct shape *is*
//!   the documented hierarchy; doc-comments explain the
//!   contained-vs-sibling distinction.

pub mod driver_cuda;
pub mod fire;
