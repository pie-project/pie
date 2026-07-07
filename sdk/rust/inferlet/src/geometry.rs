//! KV page-geometry — the token→page math the working-set refactor (W4) pushed
//! to the guest.
//!
//! Post-W4 the runtime is **token-agnostic**: it owns only physical KV page
//! slots (a dense ordered array). Everything semantic — the sequence cursor,
//! the read/write page split, the mid-page offset — is derived here, in the
//! guest. The `Context`/`Forward` facades wrap this, and the raw-WIT inferlets
//! (`generate`, `runahead`) re-inline it by hand. This module is the single
//! source of truth for that math, kept as **free functions over the raw WIT**
//! (`working-set` + `inference`) so a low-level inferlet calls it directly
//! instead of re-deriving the identical `first_write_page/total_pages/offset`
//! boilerplate. It is a genuine primitive of the minimal SDK core — not a
//! facade (see `ptir-sdk-minimization-audit`).

use crate::inference::ForwardPass;
use crate::working_set::KvWorkingSet;
use crate::Result;

/// The merged `kv-working-set` read+write descriptor for a tail write of `n`
/// new tokens at sequence cursor `seq_len` (KV `page_size` tokens per page).
///
/// The 1a-verified DISJOINT convention: **read** = the prior FULL pages
/// `[0, write_start)` (all valid); **write** = the tail pages
/// `[write_start, write_start + write_pages)`. The partial-prior prefix (when
/// `seq_len` is mid-page) plus the new tokens ride the write pages, addressed
/// by `offset`. Maps 1:1 to
/// `ForwardPass::kv_working_set(set, /*inp_start*/ 0, read_pages, valid_tokens,
/// write_start, write_pages, offset)`.
///
/// Units: `read_pages`/`write_start`/`write_pages`/`total_pages` are PAGE/slot
/// indices; `valid_tokens`/`offset` are TOKEN counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvWriteGeometry {
    /// Read page span length = `first_write_page` (the prior FULL pages).
    pub read_pages: u32,
    /// Read valid tokens = `read_pages * page_size` (every read page is full).
    pub valid_tokens: u32,
    /// Write sub-range start page = `first_write_page`.
    pub write_start: u32,
    /// Write sub-range length = `total_pages - first_write_page`.
    pub write_pages: u32,
    /// In-page token offset of the first NEW token = `seq_len % page_size`.
    pub offset: u32,
    /// Total pages needed to cover `seq_len + n` = `ceil((seq_len + n) / page_size)`.
    pub total_pages: u32,
}

/// Compute the tail-write geometry for `n` new tokens at cursor `seq_len`.
///
/// **Pure** — no host calls, no allocation. Use [`ensure_pages`] to grow the
/// page-slot array, then [`attach_kv_write`] to bind it to a forward pass.
///
/// # Panics / preconditions
/// `page_size` must be non-zero (a real KV working set always reports
/// `page_size() >= 1`).
pub fn kv_write_geometry(seq_len: u32, n: u32, page_size: u32) -> KvWriteGeometry {
    let first_write_page = seq_len / page_size;
    let total_pages = (seq_len + n).div_ceil(page_size);
    KvWriteGeometry {
        read_pages: first_write_page,
        valid_tokens: first_write_page * page_size,
        write_start: first_write_page,
        write_pages: total_pages - first_write_page,
        offset: seq_len % page_size,
        total_pages,
    }
}

/// Ensure `kv` holds at least `geom.total_pages` page slots, allocating the
/// shortfall. Returns `geom` unchanged for chaining. This is the one
/// host-touching helper — the geometry itself is pure.
pub fn ensure_pages(kv: &KvWorkingSet, geom: KvWriteGeometry) -> Result<KvWriteGeometry> {
    let have = kv.size();
    if geom.total_pages > have {
        let grow = geom.total_pages - have;
        kv.alloc(grow)
            .map_err(|e| format!("geometry::ensure_pages: alloc {grow}: {e}"))?;
    }
    Ok(geom)
}

/// Attach `geom` to `pass` as a tail-write `kv-working-set` — the 7-arg WIT
/// call spelled once (`inp_start` is always 0: the read span starts at page 0).
pub fn attach_kv_write(pass: &ForwardPass, kv: &KvWorkingSet, geom: &KvWriteGeometry) {
    pass.kv_working_set(
        kv,
        0,
        geom.read_pages,
        geom.valid_tokens,
        geom.write_start,
        geom.write_pages,
        geom.offset,
    );
}

/// Attach a **read-only** `kv-working-set` spanning all `seq_len` materialized
/// tokens (a pure decode / scoring pass that writes no tail): `valid_tokens =
/// seq_len` (the real mid-page count, since no write page carries the tail),
/// `write_pages = 0`. No-op when `seq_len == 0`.
pub fn attach_kv_read(pass: &ForwardPass, kv: &KvWorkingSet, seq_len: u32, page_size: u32) {
    let total_pages = seq_len.div_ceil(page_size);
    if total_pages > 0 {
        pass.kv_working_set(kv, 0, total_pages, seq_len, 0, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fresh sequence, sub-page write: all-write, no read context.
    #[test]
    fn fresh_subpage_write() {
        let g = kv_write_geometry(0, 3, 16);
        assert_eq!(
            g,
            KvWriteGeometry {
                read_pages: 0,
                valid_tokens: 0,
                write_start: 0,
                write_pages: 1,
                offset: 0,
                total_pages: 1,
            }
        );
    }

    // Mid-page cursor: the partial prior page is rewritten (rides the write
    // range via `offset`), read covers only the FULL prior pages.
    #[test]
    fn midpage_cursor_rewrites_partial_page() {
        // seq_len=20, page=16 → 1 full prior page [0,16); token 16..20 sit in
        // page 1 at offset 16%16=4. Writing 5 more tokens (20..25) stays in page 1.
        let g = kv_write_geometry(20, 5, 16);
        assert_eq!(g.read_pages, 1);
        assert_eq!(g.valid_tokens, 16);
        assert_eq!(g.write_start, 1);
        assert_eq!(g.write_pages, 1); // ceil(25/16)=2, minus first_write_page 1
        assert_eq!(g.offset, 4);
        assert_eq!(g.total_pages, 2);
    }

    // Page-aligned cursor: read = all prior full pages, write starts a fresh page.
    #[test]
    fn page_aligned_cursor() {
        let g = kv_write_geometry(32, 1, 16);
        assert_eq!(g.read_pages, 2);
        assert_eq!(g.valid_tokens, 32);
        assert_eq!(g.write_start, 2);
        assert_eq!(g.write_pages, 1); // ceil(33/16)=3 - 2
        assert_eq!(g.offset, 0);
        assert_eq!(g.total_pages, 3);
    }

    // Write spanning a page boundary from a full-page cursor.
    #[test]
    fn write_spans_page_boundary() {
        let g = kv_write_geometry(16, 20, 16);
        assert_eq!(g.read_pages, 1);
        assert_eq!(g.write_start, 1);
        assert_eq!(g.write_pages, 2); // ceil(36/16)=3 - 1
        assert_eq!(g.offset, 0);
        assert_eq!(g.total_pages, 3);
    }
}
