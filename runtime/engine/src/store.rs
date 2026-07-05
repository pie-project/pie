//! Typed resource stores (kv_refact.md).
//!
//! Replaces the generic KV-page-sized `Arena` with resource-specific stores
//! over typed static backing pools. This module currently hosts the pure host
//! core: the physical-id free list (`pool`), semantic hashing (`kv::hash`),
//! and the KV mapping trie (`kv::page_table`). Wiring into WIT resources,
//! the PTIR fire path, and driver flat-table publication lands in later
//! increments; `arena/` and `working_set/` remain authoritative until then.

pub mod genmap;
pub mod kv;
pub mod pool;
pub mod registry;
pub mod rs;
