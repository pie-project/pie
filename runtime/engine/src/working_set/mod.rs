//! Working-set runtime cores: dense KV page arrays (`kv`) and recurrent-state
//! folded+buffered sets (`rs`), backed by the unified typed arena. These are
//! the token-agnostic memory cores behind the WIT `kv-working-set` /
//! `rs-working-set` resources (W2/W4/W8). The host bindings live under
//! `crate::api`.
//!
//! North star ([[workingset-brief]]): *inferlet owns meaning, working sets own
//! memory references, runtime owns physical feasibility*. The only thing
//! crossing the inferlet API is a relative `u32` index — never a physical page
//! id or vpage id (W2). The physical layer is bravo's unified arena
//! (`crate::arena`): KV pages are `ArenaHandle{KvPage}`, RS slabs are
//! `ArenaHandle{RsSlab}`, with arena-owned refcounts + lazy CoW.
//!
//! `kv_cas` is the per-(model,driver) content-addressed sharing index for KV
//! pages (charlie's lane), a sibling registry to `crate::arena::registry`.

pub mod kv;
pub mod kv_cas;
pub mod page_hash;
pub mod page_size;
pub mod rs;
