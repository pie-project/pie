//! Fire KV preparation over the typed `KvStore` (kv_refact.md), plus the
//! canonical-KV shape/evidence gate and the WIT-descriptor validation
//! (`PrepareError` re-export, `check_generation`/`check_input_nonempty`).
//!
//! `pipeline::fire::submit` calls [`prepare`] to classify the fire's KV
//! write intents (fresh append / private in-place / shared-tail CoW rebase),
//! allocate physical pages, and project the driver page geometry; it threads
//! the returned [`KvTxn`] across the async fire and
//! [`finalize`]s it (commit publishes the mapping; abort releases the
//! pending slots and leaves the committed mapping authoritative).
//!
//! Hash lifecycle (increment 1): canonical fires (bind-time shape gate
//! [`canonical_kv_shape`] + fire-time host-known-token gate
//! [`canonical_fire_evidence`], both called from `pipeline::fire`) commit
//! chained `(token, position)` slot hashes and full-page hashes — feeding the
//! store's chain state and CAS index; every other fire commits opaque slot
//! hashes (concrete identity, never matchable). Matching/trim is the next
//! increment.
//!
//! Complete pipeline domain API: some methods here (relaxed geometry
//! variants, per-channel introspection, the pure `instantiate`/registry
//! probe entry points, device-geometry lease internals) are not yet
//! called by the current single-model/mock-driver fire path, but are
//! exercised by this module's own unit tests and reserved for upcoming
//! wiring (multi-pass channels, device-geometry beams) — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use crate::store::kv::hash::{self, Hash256};
use crate::store::kv::page_table::WorkingSetId;
pub use crate::store::kv::project::PrepareError;
use crate::store::kv::project::{KvProjection, KvWrite, project_kv};
use crate::store::kv::write::{KvPreparedWrite, PageCommit, PreparedTarget};
use crate::store::kv::{KvStore, KvStoreError};

/// A KV prepare failure. Pool exhaustion stays typed so the fire path can
/// route it through the contention ladder (acquire, then RETRY the prepare);
/// everything else is a guest-visible fire error.
#[derive(Debug)]
pub enum KvError {
    /// The physical pool could not supply `requested` pages. Retryable after
    /// the ladder frees pages (`available` is the shortfall context).
    OutOfPages {
        requested: usize,
        available: usize,
    },
    Fatal(String),
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvError::OutOfPages {
                requested,
                available,
            } => write!(
                f,
                "kv pool exhausted: requested {requested}, available {available}"
            ),
            KvError::Fatal(e) => f.write_str(e),
        }
    }
}

impl From<KvStoreError> for KvError {
    fn from(e: KvStoreError) -> Self {
        match e {
            KvStoreError::OutOfPages {
                requested,
                available,
            } => KvError::OutOfPages {
                requested,
                available,
            },
            other => KvError::Fatal(other.to_string()),
        }
    }
}

/// The prepared KV write for one in-flight PTIR fire — held across
/// `submit_async` until [`finalize`].
pub struct KvTxn {
    seq: u64,
    cas_intents: Vec<crate::store::kv::CasIntent>,
    mapping_version: u64,
}

impl KvTxn {
    pub fn mapping_version(&self) -> u64 {
        self.mapping_version
    }
}

/// The fire's WorkingSet page translation (kv_refact.md flattened-table
/// model): entry `i` = the physical page backing WS-relative index `i`,
/// i.e. the committed flat table overlaid with this fire's prepared write
/// targets. Ships with the launch so the driver can map channel-resolved
/// `Pages`/`WSlot` references; guests only ever hold relative indexes.
fn build_translation(
    store: &mut KvStore,
    ws: WorkingSetId,
) -> Result<(u64, Vec<u32>), KvStoreError> {
    let (version, table) = store.flat_table(ws)?;
    Ok((version, table.iter().map(|page| page.0).collect()))
}

/// Build the per-target [`PageCommit`]s for a fire appending `n_new` tokens
/// at `append_start`. `hash_tokens = Some(values)` on a canonical fire
/// (bind-time shape + fire-time host-known gate both passed): the new slots
/// chain `(token, position)` identities from the WorkingSet's chain state,
/// and pages that come out FULL get a page hash (which marks them for the
/// CAS index). Otherwise every written slot draws an opaque hash — concrete
/// identity that survives forks/selections but never matches anything.
/// Preserved slots (in-place prefix, unwritten CoW-copied pages) carry their
/// existing hashes; an unwritten CoW page also keeps its page hash (the copy
/// preserves content).
fn build_commits(
    store: &mut KvStore,
    prepared: &KvPreparedWrite,
    ws: WorkingSetId,
    append_start: u32,
    n_new: u32,
    page_size: u32,
    hash_tokens: Option<&[u32]>,
) -> Result<Vec<PageCommit>, KvStoreError> {
    let canonical = hash_tokens.is_some();
    let domain = store.domain();
    let mut prev = store.chain_state(ws)?;
    let mut slot_hashes: Vec<Hash256> = Vec::with_capacity(n_new as usize);
    for j in 0..n_new {
        let h = match hash_tokens {
            Some(tokens) => hash::chain_token_slot_hash(
                &domain,
                prev.as_ref(),
                tokens[j as usize],
                append_start + j,
            ),
            None => store.next_opaque_hash(),
        };
        prev = Some(h);
        slot_hashes.push(h);
    }

    let mut commits = Vec::with_capacity(prepared.targets().len());
    for target in prepared.targets() {
        let page = target.index();
        let (mut hashes, existing_page_hash) = match target {
            PreparedTarget::Fresh { .. } => (Vec::new(), None),
            PreparedTarget::InPlace { index, .. } | PreparedTarget::Cow { index, .. } => (
                store.page_token_hashes(ws, *index)?,
                store.page_hash_at(ws, *index)?,
            ),
        };
        hashes.resize(page_size as usize, None);

        // Written slots of this page: global token indexes
        // [append_start, append_start + n_new) landing on page `page`.
        let mut wrote = false;
        for (j, h) in slot_hashes.iter().enumerate() {
            let tok = append_start as u64 + j as u64;
            if tok / page_size as u64 == page {
                hashes[(tok % page_size as u64) as usize] = Some(*h);
                wrote = true;
            }
        }

        let page_hash = if !wrote {
            existing_page_hash // pure CoW copy: content (and identity) preserved
        } else if canonical && hashes.iter().all(|h| h.is_some()) {
            Some(hash::page_hash(&hashes))
        } else {
            None
        };
        commits.push(PageCommit {
            token_hashes: hashes,
            page_hash,
        });
    }
    Ok(commits)
}

/// Realize the mapped overlap of one writable declaration exactly once.
/// Shared pages rebase through the existing COW protocol; private pages only
/// lose transitional implicit-cache identity. Fresh backing is handled
/// separately by [`KvStore::ensure_backed`].
pub fn realize_declaration(
    store: &mut KvStore,
    ws: WorkingSetId,
    writable: std::ops::Range<u64>,
) -> Result<((Vec<u32>, Vec<u32>), Option<KvTxn>), KvError> {
    let mapped = store.mapped_len(ws)?;
    let start = writable.start.min(mapped);
    let end = writable.end.min(mapped);
    if start >= end {
        return Ok(((Vec::new(), Vec::new()), None));
    }
    let indexes: Vec<u64> = (start..end).collect();
    let shared = indexes
        .iter()
        .copied()
        .map(|index| store.privately_writable(ws, index))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .any(|private| !private);
    if !shared {
        store.opacify_suffix(ws, start)?;
        return Ok(((Vec::new(), Vec::new()), None));
    }

    let prepared = store.prepare_write(ws, &indexes)?;
    let commits = prepared
        .targets()
        .iter()
        .map(|target| {
            let index = target.index();
            Ok(PageCommit {
                token_hashes: store.page_token_hashes(ws, index)?,
                page_hash: store.page_hash_at(ws, index)?,
            })
        })
        .collect::<Result<Vec<_>, KvStoreError>>()?;
    let copies = prepared
        .copy_plan()
        .map(|(src, dst)| (src.0, dst.0))
        .unzip();
    let (seq, cas_intents) = store.publish_prepared(prepared, &commits)?;
    store.opacify_suffix(ws, start)?;
    let (mapping_version, _) = build_translation(store, ws)?;
    Ok((
        copies,
        Some(KvTxn {
            seq,
            cas_intents,
            mapping_version,
        }),
    ))
}

/// Empty-WS prefill prefix match (kv_refact.md "Trie Matching", increment
/// 2a): probe the CAS index for the LONGEST full-page prefix of `tokens`
/// already resident, and graft it into `ws` on a hit. Always leaves at least
/// one token to compute (the readout row must run), so
/// `matched * page_size < tokens.len()`. After a hit the caller prepares the
/// fire as a continuation (`committed = matched * page_size`, the token
/// suffix as `new_tokens`) — which additionally requires the
/// descriptor-level pass trim on the driver side, the next increment; until
/// then this is exercised by the store tests only.
pub fn match_prefix(
    store: &mut KvStore,
    ws: WorkingSetId,
    tokens: &[u32],
    page_size: u32,
) -> Result<Option<u64>, KvError> {
    if store.mapped_len(ws)? != 0 || store.chain_state(ws)?.is_some() {
        return Ok(None); // only a fresh, never-written working set
    }
    let ps = page_size as usize;
    let max_pages = tokens.len().saturating_sub(1) / ps;
    if max_pages == 0 {
        return Ok(None);
    }
    // Boundary chain values at each candidate full-page boundary — the same
    // chain a canonical prefill of these tokens would commit.
    let domain = store.domain();
    let mut prev: Option<Hash256> = None;
    let mut boundaries = Vec::with_capacity(max_pages);
    for (i, &tok) in tokens[..max_pages * ps].iter().enumerate() {
        let h = hash::chain_token_slot_hash(&domain, prev.as_ref(), tok, i as u32);
        prev = Some(h);
        if (i + 1) % ps == 0 {
            boundaries.push(h);
        }
    }
    for pages in (1..=max_pages).rev() {
        let key = boundaries[pages - 1];
        if let Some(adopted) = store.adopt_cached_prefix(ws, &key, pages as u64)? {
            return Ok(Some(adopted));
        }
    }
    Ok(None)
}

/// Prepare the KV projection for a PTIR fire appending `new_tokens` to `ws`
/// at the explicit `append_start`.
///
/// Returns `(proj, (copy_src, copy_dst), txn)`: pass
/// `proj.physical_page_ids` / `proj.last_page_len` into `submit_async`, issue
/// one `scheduler::copy_d2d(copy_src, copy_dst)` for the CoW-preserved pages
/// before the launch when non-empty, hold `txn` across the fire, then
/// [`finalize`]. `new_tokens`' VALUES are unused for the projection
/// (pure page geometry keyed by the count); `hash_tokens = Some(values)` is
/// the canonical-fire gate — the HOST-VERIFIED token values this fire embeds
/// (see `canonical_kv_shape` + the host-known gate in `submit_pass`), which
/// the committed pages hash under. `None` ⇒ opaque slot hashes.
pub fn prepare(
    store: &mut KvStore,
    ws: WorkingSetId,
    append_start: u32,
    new_tokens: &[u32],
    page_size: u32,
    hash_tokens: Option<&[u32]>,
) -> Result<(KvProjection, (Vec<u32>, Vec<u32>), Vec<u32>, KvTxn), KvError> {
    prepare_impl(store, ws, append_start, new_tokens, page_size, hash_tokens)
}

fn prepare_impl(
    store: &mut KvStore,
    ws: WorkingSetId,
    append_start: u32,
    new_tokens: &[u32],
    page_size: u32,
    hash_tokens: Option<&[u32]>,
) -> Result<(KvProjection, (Vec<u32>, Vec<u32>), Vec<u32>, KvTxn), KvError> {
    debug_assert!(hash_tokens.is_none_or(|t| t.len() == new_tokens.len()));
    let n_new = new_tokens.len() as u32;
    if n_new == 0 {
        return Err(KvError::Fatal(
            "prepare: new_tokens must be non-empty".to_string(),
        ));
    }

    let total = append_start + n_new;
    let needed_pages = total.div_ceil(page_size) as u64;

    // Grow the logical address space so every write slot exists. Purely
    // logical: physical pages are allocated by prepare_write below.
    let page_len = store.page_len(ws)?;
    if page_len < needed_pages {
        store.reserve(ws, needed_pages - page_len)?;
    }

    // Prior context: pages [0, valid_pages) for the committed tokens, from
    // the flattened table (write targets override their slots below).
    let valid_pages = (append_start.div_ceil(page_size)) as usize;
    let context_pages: Vec<u32> = {
        store
            .flat_table(ws)?
            .1
            .iter()
            .copied()
            .into_iter()
            .take(valid_pages)
            .map(|p| p.0)
            .collect()
    };
    if context_pages.len() < valid_pages {
        return Err(KvError::Fatal(format!(
            "prepare: committed {append_start} tokens but only {} mapped pages",
            context_pages.len()
        )));
    }

    // Classify + allocate the write slots [output_start, needed_pages).
    // `KvStoreError::OutOfPages` stays typed through here — the caller
    // routes it into the contention ladder and retries.
    let output_start = (append_start / page_size) as u64;
    let write_indexes: Vec<u64> = (output_start..needed_pages).collect();
    let prepared = store.prepare_write(ws, &write_indexes)?;

    // Driver geometry: every prepared target is a written slot (the CoW
    // rebase never reaches below the first written committed page).
    let offset = append_start % page_size;
    let writes: Vec<KvWrite> = prepared
        .targets()
        .iter()
        .map(|t| {
            let slot = t.index() as u32;
            let i = slot.saturating_sub(output_start as u32);
            let valid_len = (offset + n_new)
                .saturating_sub(i * page_size)
                .min(page_size);
            KvWrite {
                slot_index: slot,
                page: t.dst().0,
                valid_len,
            }
        })
        .collect();

    let (copy_src, copy_dst): (Vec<u32>, Vec<u32>) =
        prepared.copy_plan().map(|(s, d)| (s.0, d.0)).unzip();

    let proj = match project_kv(&context_pages, append_start, &writes, page_size) {
        Ok(projection) => projection,
        Err(error) => {
            let targets = prepared
                .targets()
                .iter()
                .map(|target| target.index())
                .collect::<Vec<_>>();
            store.cancel_prepared(prepared);
            return Err(KvError::Fatal(format!(
                "{error:?} (committed={append_start}, new={n_new}, targets={targets:?})"
            )));
        }
    };

    let commits = match build_commits(
        store,
        &prepared,
        ws,
        append_start,
        n_new,
        page_size,
        hash_tokens,
    ) {
        Ok(commits) => commits,
        Err(error) => {
            store.cancel_prepared(prepared);
            return Err(error.into());
        }
    };
    let (seq, cas_intents) = store.publish_prepared(prepared, &commits)?;
    let (translation_version, translation) = match build_translation(store, ws) {
        Ok(translation) => translation,
        Err(error) => {
            store.settle(cas_intents, false);
            store.retire_through(seq);
            return Err(error.into());
        }
    };

    Ok((
        proj,
        (copy_src, copy_dst),
        translation,
        KvTxn {
            seq,
            cas_intents,
            mapping_version: translation_version,
        },
    ))
}

/// Prepare an explicit-KV (device-geometry) fire: physical pages for
/// `write_indexes` with no host projection — the driver resolves the geometry
/// itself and the inferlet owns the token bookkeeping. Returns the
/// `(index, physical id)` pairs for the granted slots, the CoW copy plan, and
/// the held transaction.
pub fn prepare_explicit(
    store: &mut KvStore,
    ws: WorkingSetId,
    write_indexes: &[u64],
) -> Result<(Vec<(u64, u32)>, (Vec<u32>, Vec<u32>), Vec<u32>, KvTxn), KvError> {
    let prepared = store.prepare_write(ws, write_indexes)?;
    let pages: Vec<(u64, u32)> = prepared
        .targets()
        .iter()
        .map(|t| (t.index(), t.dst().0))
        .collect();
    let copies = prepared.copy_plan().map(|(s, d)| (s.0, d.0)).unzip();
    // Device-geometry fires are non-canonical by construction, and the
    // device owns the token bookkeeping — commit with no hash metadata;
    // `KvStore::commit` poisons the chain state with an opaque draw.
    let commits: Vec<PageCommit> = prepared
        .targets()
        .iter()
        .map(|_| PageCommit {
            token_hashes: Vec::new(),
            page_hash: None,
        })
        .collect();
    let (seq, cas_intents) = store.publish_prepared(prepared, &commits)?;
    let (translation_version, translation) = match build_translation(store, ws) {
        Ok(translation) => translation,
        Err(error) => {
            store.settle(cas_intents, false);
            store.retire_through(seq);
            return Err(error.into());
        }
    };
    Ok((
        pages,
        copies,
        translation,
        KvTxn {
            seq,
            cas_intents,
            mapping_version: translation_version,
        },
    ))
}

/// Settle a prepared fire as failed. Its mapping remains pipeline-local
/// fail-stop state; CAS publication is discarded.
pub fn abandon(store: &mut KvStore, txn: KvTxn) {
    let KvTxn {
        seq, cas_intents, ..
    } = txn;
    store.settle(cas_intents, false);
    store.retire_through(seq);
}

/// Finalize a PTIR fire's KV write after `submit_async` resolves. `success`
/// publishes the mapping (pages persist for the next fire); otherwise the
/// pending slots release and the committed mapping is untouched. Fires retire
/// in FIFO stream order, so this fire's sequence retires every recycle tagged
/// at or before it.
pub fn finalize(store: &mut KvStore, txn: KvTxn, success: bool) -> Result<(), String> {
    let KvTxn {
        seq, cas_intents, ..
    } = txn;
    store.settle(cas_intents, success);
    store.retire_through(seq);
    Ok(())
}

/// Reject a forward with no query rows. `num_input_rows` is the driver
/// `token_ids` length — text tokens plus the placeholder rows that image/audio
/// spans contribute — i.e. the span of `qo_indptr`. A pass must compute at
/// least one row; this mirrors the old context API's "must supply at least one
/// token" invariant (W4: input lineage is inferlet-owned, so the runtime can no
/// longer infer a token from an ambient context).
pub fn check_input_nonempty(num_input_rows: usize) -> Result<(), PrepareError> {
    if num_input_rows == 0 {
        Err(PrepareError::NoInputTokens)
    } else {
        Ok(())
    }
}

/// Validate a captured generation against the working set's current one.
/// Called first in prepare so a stale mutation fails before any arena work.
pub fn check_generation(captured: u32, current: u32) -> Result<(), PrepareError> {
    if captured == current {
        Ok(())
    } else {
        Err(PrepareError::StaleGeneration { captured, current })
    }
}

/// Bind-time half of the canonical-KV gate (kv_refact.md, "Token-Slot
/// Hashes, Page Hashes, and Trie Matching"): the pass writes exactly what
/// the vanilla model produces for one appended token run under full causal
/// self-attention over the working set — so its KV rows may carry chained
/// semantic hashes. Rejected by anything that can perturb K/V production:
/// an attention mask (it changes hidden states, hence KV at layers > 0),
/// per-layer stage programs (they can rewrite projections), or extern
/// channels. Prologue/epilogue programs only shape sampling — grammar,
/// watermarking, and sampler passes all stay canonical. A `KvLen` port must
/// exist so the fire-time gate can verify the pass attends the FULL context
/// (a shorter span changes upper-layer KV).
///
/// KvLen-root dense defaults are canonical when any author-bound overrides
/// agree with the same contiguous append. A channel-fed `EmbedIndptr`
/// (dynamic lane structure) still rejects; const CSRs are value-checked at
/// fire time.
pub fn canonical_kv_shape(container: &pie_ptir::container::TraceContainer) -> bool {
    use pie_ptir::container::PortSource;
    use pie_ptir::registry::{Port, Stage};

    if !container.externs.is_empty() {
        return false;
    }
    let mut has_kv_len = false;
    for binding in &container.ports {
        match binding.port {
            Port::AttnMask => return false,
            Port::KvLen => has_kv_len = true,
            Port::EmbedIndptr => {
                if matches!(binding.source, PortSource::Channel(_)) {
                    return false;
                }
            }
            _ => {}
        }
    }
    has_kv_len
        && !container
            .stages
            .iter()
            .any(|s| matches!(s.stage, Stage::OnAttnProj | Stage::OnAttn))
}

pub struct CanonicalFireEvidence {
    tokens: Vec<u32>,
    /// Per-lane attended context (one entry for a single-lane fire, one per
    /// token for the per-token CSR form).
    kv_len: Vec<u32>,
    embed_indptr: Option<Vec<u32>>,
    positions: Option<Vec<u32>>,
    /// Wire-form lane page CSR (a rank-2 `[lanes, P]` envelope arrives
    /// already compacted by the evaluated-geometry mapper).
    pages: Option<Vec<u32>>,
    page_indptr: Option<Vec<u32>>,
    w_slot: Option<Vec<u32>>,
    w_off: Option<Vec<u32>>,
}

/// Fire-time half of the canonical-KV gate, over the fire's **evaluated
/// descriptor ports** (`pie_ptir::pareval` folded through the host shadow):
/// the host-verified token values this fire embeds, the kv-len it attends,
/// and the append geometry the host derived. `None` unless the bind-time
/// shape passed ([`canonical_kv_shape`], captured as `canonical_kv` at bind)
/// and the embed/kv-len values are host-known — a device-decided value
/// (loop-carried sampler output past the seed fire) yields no evidence, it
/// never guesses. [`canonical_hash_tokens`] then verifies the values form
/// this fire's contiguous full-context append.
pub fn canonical_fire_evidence(
    canonical_kv: bool,
    ports: &[(
        pie_ptir::registry::Port,
        Result<pie_ptir::interp::Value, String>,
    )],
    container: &pie_ptir::container::TraceContainer,
) -> Option<CanonicalFireEvidence> {
    use pie_ptir::registry::Port;

    if !canonical_kv {
        return None;
    }
    let get = |port: Port| ports.iter().find(|(p, _)| *p == port);
    let known = |port: Port| -> Option<Vec<u32>> {
        match get(port) {
            Some((_, Ok(value))) => Some(super::geometry::value_as_u32(value)),
            _ => None,
        }
    };
    // Unbound is fine (`Some(None)`); bound-but-unknown kills the evidence.
    let optional = |port: Port| -> Option<Option<Vec<u32>>> {
        match get(port) {
            None => Some(None),
            Some((_, Ok(value))) => Some(Some(super::geometry::value_as_u32(value))),
            Some((_, Err(_))) => None,
        }
    };
    let page_indptr = optional(pie_ptir::registry::Port::PageIndptr)?;
    let pages = match optional(Port::Pages)? {
        Some(raw) => Some(
            super::geometry::compact_page_envelope(
                container,
                raw,
                page_indptr.as_deref().unwrap_or(&[]),
            )
            .ok()?,
        ),
        None => None,
    };
    Some(CanonicalFireEvidence {
        tokens: known(Port::EmbedTokens)?,
        kv_len: known(Port::KvLen)?,
        embed_indptr: optional(Port::EmbedIndptr)?,
        positions: optional(Port::Positions)?,
        pages,
        page_indptr,
        w_slot: optional(Port::WSlot)?,
        w_off: optional(Port::WOff)?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalAppend {
    pub start: u32,
    pub tokens: Vec<u32>,
}

/// Verify the evidence forms a canonical contiguous append. `KvLen` is the
/// root: it determines the appended span, and explicit positions/write
/// geometry must agree when present.
pub fn canonical_hash_tokens(
    evidence: CanonicalFireEvidence,
    request: &crate::driver::LaunchPlan,
    device_resolved: bool,
    page_size: u32,
) -> Option<CanonicalAppend> {
    let n = evidence.tokens.len();
    if n == 0 || page_size == 0 || evidence.tokens.iter().any(|&t| t == u32::MAX) {
        return None;
    }

    // Lane structure: one lane over all tokens, or one token per lane.
    let per_token = match &evidence.embed_indptr {
        None => false,
        Some(v) if v.as_slice() == [0, n as u32] => false,
        Some(v) if v.len() == n + 1 && v.iter().enumerate().all(|(i, &x)| x == i as u32) => true,
        Some(_) => return None,
    };
    let lanes = if per_token { n } else { 1 };
    if evidence.kv_len.len() != lanes {
        return None;
    }
    let start = if per_token {
        evidence.kv_len.first().copied()?.checked_sub(1)?
    } else {
        evidence.kv_len[0].checked_sub(n as u32)?
    };
    let end = start.checked_add(n as u32)?;

    // Positions: the contiguous append span (absent = driver append order).
    if let Some(positions) = &evidence.positions
        && (positions.len() != n
            || positions
                .iter()
                .enumerate()
                .any(|(i, &p)| p != start + i as u32))
    {
        return None;
    }

    // Full-context attention per lane.
    for lane in 0..lanes {
        let expected = if per_token {
            start + lane as u32 + 1
        } else {
            end
        };
        if evidence.kv_len[lane] != expected {
            return None;
        }
    }

    let mut default_pages = Vec::new();
    let mut default_indptr = Vec::with_capacity(lanes + 1);
    default_indptr.push(0);
    for &len in &evidence.kv_len {
        default_pages.extend(0..len.div_ceil(page_size));
        default_indptr.push(u32::try_from(default_pages.len()).ok()?);
    }
    let pages = evidence.pages.as_deref().unwrap_or(&default_pages);
    let page_indptr = evidence.page_indptr.as_deref().unwrap_or(&default_indptr);
    if page_indptr.len() != lanes + 1 || page_indptr[0] != 0 {
        return None;
    }
    for lane in 0..lanes {
        let (start, end) = (page_indptr[lane] as usize, page_indptr[lane + 1] as usize);
        if end < start || end > pages.len() {
            return None;
        }
        let lane_pages = &pages[start..end];
        let required = evidence.kv_len[lane].div_ceil(page_size) as usize;
        if required == 0
            || required > lane_pages.len()
            || lane_pages[..required]
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != required
        {
            return None;
        }
    }
    for index in 0..n {
        let lane = if per_token { index } else { 0 };
        let (page_start, page_end) = (page_indptr[lane] as usize, page_indptr[lane + 1] as usize);
        let lane_pages = &pages[page_start..page_end];
        let position = start + index as u32;
        let page = (position / page_size) as usize;
        let expected_slot = *lane_pages.get(page)?;
        if evidence
            .w_slot
            .as_ref()
            .is_some_and(|slots| slots.get(index) != Some(&expected_slot))
            || evidence
                .w_off
                .as_ref()
                .is_some_and(|offsets| offsets.get(index) != Some(&(position % page_size)))
        {
            return None;
        }
    }

    // Wire agreement — the driver must execute exactly what we hash. A
    // device-resolved fire executes from the channel state the evaluator
    // mirrored instead of the wire (the classifier parity corpus pins the
    // two resolutions together).
    if !device_resolved {
        if evidence.tokens != request.token_ids {
            return None;
        }
        if let Some(positions) = &evidence.positions
            && *positions != request.position_ids
        {
            return None;
        }
    }
    Some(CanonicalAppend {
        start,
        tokens: evidence.tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonce() -> [u8; 32] {
        [7u8; 32]
    }

    use pie_ptir::container::{ChanDType, ChannelDecl, HostRole, PortBinding, StageProgram};
    use pie_ptir::registry::{Port, Stage};
    use pie_ptir::types::{DType, Shape};

    fn ch(shape: Shape, dtype: DType, role: HostRole) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded: false,
        }
    }

    /// A minimal canonical decode container: embed tokens + kv-len + the
    /// explicit append geometry every SDK-lowered pass carries (RV-14) +
    /// epilogue. Channels: 0 tok (device-loop), 1 klen, 2 pages,
    /// 3 page-indptr, 4 w_slot, 5 w_off.
    fn plain_decode_container() -> pie_ptir::container::TraceContainer {
        pie_ptir::container::TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::vector(1), DType::I32, HostRole::None),
                ch(Shape::vector(1), DType::U32, HostRole::None),
                ch(Shape::vector(4), DType::U32, HostRole::None),
                ch(Shape::vector(2), DType::U32, HostRole::None),
                ch(Shape::vector(1), DType::U32, HostRole::None),
                ch(Shape::vector(1), DType::U32, HostRole::None),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: pie_ptir::container::PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::KvLen,
                    source: pie_ptir::container::PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::Pages,
                    source: pie_ptir::container::PortSource::Channel(2),
                },
                PortBinding {
                    port: Port::PageIndptr,
                    source: pie_ptir::container::PortSource::Channel(3),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: pie_ptir::container::PortSource::Channel(4),
                },
                PortBinding {
                    port: Port::WOff,
                    source: pie_ptir::container::PortSource::Channel(5),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
            externs: vec![],
        }
    }

    #[test]
    fn canonical_shape_accepts_the_plain_decode() {
        assert!(canonical_kv_shape(&plain_decode_container()));
    }

    #[test]
    fn canonical_shape_accepts_kv_len_root_defaults() {
        for port in [Port::Pages, Port::PageIndptr, Port::WSlot, Port::WOff] {
            let mut c = plain_decode_container();
            c.ports.retain(|p| p.port != port);
            assert!(
                canonical_kv_shape(&c),
                "a container without {port:?} uses the KvLen-root dense default"
            );
        }
    }

    #[test]
    fn canonical_shape_rejects_kv_perturbing_passes() {
        // Attention mask: changes hidden states, hence KV at layers > 0.
        let mut c = plain_decode_container();
        c.ports.push(PortBinding {
            port: Port::AttnMask,
            source: pie_ptir::container::PortSource::Channel(0),
        });
        assert!(!canonical_kv_shape(&c));

        // Per-layer stage program: can rewrite the projections.
        let mut c = plain_decode_container();
        c.stages.push(StageProgram {
            stage: Stage::OnAttn,
            ops: vec![],
        });
        assert!(!canonical_kv_shape(&c));

        // No KvLen port: the full-context claim cannot be verified.
        let mut c = plain_decode_container();
        c.ports.retain(|p| p.port != Port::KvLen);
        assert!(!canonical_kv_shape(&c));

        // Device geometry is inferlet-managed layout (WSlot/WOff write
        // descriptors + a [B,P] Pages channel — see
        // `pipeline::fire::lease::detect_device_geometry`).
        let devgeo = pie_ptir::container::TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::matrix(2, 3), DType::U32, HostRole::None), // pages
                ch(Shape::vector(2), DType::U32, HostRole::None),    // w_slot
                ch(Shape::vector(2), DType::U32, HostRole::None),    // w_off
            ],
            ports: vec![
                PortBinding {
                    port: Port::Pages,
                    source: pie_ptir::container::PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: pie_ptir::container::PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::WOff,
                    source: pie_ptir::container::PortSource::Channel(2),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
            externs: vec![],
        };
        assert!(!canonical_kv_shape(&devgeo));
    }

    #[test]
    fn canonical_shape_gates_embed_indptr_to_a_single_const_lane() {
        // Const [0, n] single-lane CSR: canonical.
        let mut c = plain_decode_container();
        c.ports.push(PortBinding {
            port: Port::EmbedIndptr,
            source: pie_ptir::container::PortSource::Const {
                dtype: DType::U32,
                shape: Shape::vector(2),
                data: [0u32.to_le_bytes(), 4u32.to_le_bytes()].concat(),
            },
        });
        assert!(canonical_kv_shape(&c));

        // Channel-fed indptr (dynamic lanes): not canonical.
        let mut c = plain_decode_container();
        c.ports.push(PortBinding {
            port: Port::EmbedIndptr,
            source: pie_ptir::container::PortSource::Channel(1),
        });
        assert!(!canonical_kv_shape(&c));
    }

    fn explicit_single_lane_evidence(tokens: &[u32], committed: u32) -> CanonicalFireEvidence {
        let n = tokens.len() as u32;
        CanonicalFireEvidence {
            tokens: tokens.to_vec(),
            kv_len: vec![committed + n],
            embed_indptr: Some(vec![0, n]),
            positions: Some((committed..committed + n).collect()),
            pages: Some(vec![4, 9, 12]),
            page_indptr: Some(vec![0, 3]),
            w_slot: Some(
                (committed..committed + n)
                    .map(|position| [4, 9, 12][(position / 16) as usize])
                    .collect(),
            ),
            w_off: Some(
                (committed..committed + n)
                    .map(|position| position % 16)
                    .collect(),
            ),
        }
    }

    #[test]
    fn canonical_explicit_prefill_requires_contiguous_resolved_writes() {
        let tokens = (1..=17).collect::<Vec<_>>();
        let mut request = crate::driver::LaunchPlan {
            token_ids: tokens.clone(),
            position_ids: (0..17).collect(),
            qo_indptr: vec![0, 17],
            ..crate::driver::LaunchPlan::default()
        };
        let evidence = explicit_single_lane_evidence(&tokens, 0);
        assert_eq!(
            canonical_hash_tokens(evidence, &request, false, 16),
            Some(CanonicalAppend {
                start: 0,
                tokens: tokens.clone(),
            })
        );

        // Wire disagreement (host geometry differs from evidence): no hash.
        request.position_ids[16] = 0;
        let invalid = explicit_single_lane_evidence(&tokens, 0);
        assert!(canonical_hash_tokens(invalid, &request, false, 16).is_none());
    }

    #[test]
    fn canonical_continuation_prefill_hashes_against_committed_span() {
        // 8 appended tokens over 16 already committed: evidence must check
        // against [16, 24), not [0, 8) — the old gate's committed==0 bail is
        // exactly the prefix-cache regression (RV-1).
        let tokens = (100..108).collect::<Vec<_>>();
        let request = crate::driver::LaunchPlan {
            token_ids: tokens.clone(),
            position_ids: (16..24).collect(),
            qo_indptr: vec![0, 8],
            ..crate::driver::LaunchPlan::default()
        };
        let evidence = explicit_single_lane_evidence(&tokens, 16);
        assert_eq!(
            canonical_hash_tokens(evidence, &request, false, 16),
            Some(CanonicalAppend {
                start: 16,
                tokens: tokens.clone(),
            })
        );

        // A stale evidence span (positions from 0) must not hash.
        let stale = explicit_single_lane_evidence(&tokens, 0);
        assert!(canonical_hash_tokens(stale, &request, false, 16).is_none());
    }

    #[test]
    fn canonical_per_token_csr_hashes_like_single_lane() {
        // The SDK lowering's one-token-per-lane CSR: lane i attends its own
        // causal prefix [0, i+1). Union is the contiguous [0, n) append.
        let n = 5u32;
        let tokens: Vec<u32> = (1..=n).collect();
        let pages: Vec<u32> = (0..n).flat_map(|_| [7u32]).collect(); // lane i -> page 7
        let request = crate::driver::LaunchPlan {
            token_ids: tokens.clone(),
            position_ids: (0..n).collect(),
            qo_indptr: (0..=n).collect(),
            ..crate::driver::LaunchPlan::default()
        };
        let evidence = CanonicalFireEvidence {
            tokens: tokens.clone(),
            kv_len: (1..=n).collect(),
            embed_indptr: Some((0..=n).collect()),
            positions: Some((0..n).collect()),
            pages: Some(pages),
            page_indptr: Some((0..=n).collect()),
            w_slot: Some(vec![7; n as usize]),
            w_off: Some((0..n).collect()),
        };
        assert_eq!(
            canonical_hash_tokens(evidence, &request, false, 16),
            Some(CanonicalAppend {
                start: 0,
                tokens: tokens.clone(),
            })
        );

        // A lane that under-attends its causal prefix must not hash.
        let mut short = CanonicalFireEvidence {
            tokens: tokens.clone(),
            kv_len: (1..=n).collect(),
            embed_indptr: Some((0..=n).collect()),
            positions: Some((0..n).collect()),
            pages: Some(vec![7; n as usize]),
            page_indptr: Some((0..=n).collect()),
            w_slot: Some(vec![7; n as usize]),
            w_off: Some((0..n).collect()),
        };
        short.kv_len[3] = 3; // lane 3 attends 3 < 4
        assert!(canonical_hash_tokens(short, &request, false, 16).is_none());
    }

    #[test]
    fn canonical_per_token_csr_derives_continuation_start() {
        let n = 5u32;
        let tokens: Vec<u32> = (100..100 + n).collect();
        let request = crate::driver::LaunchPlan {
            token_ids: tokens.clone(),
            position_ids: (16..16 + n).collect(),
            qo_indptr: (0..=n).collect(),
            ..crate::driver::LaunchPlan::default()
        };
        let evidence = CanonicalFireEvidence {
            tokens: tokens.clone(),
            kv_len: (17..17 + n).collect(),
            embed_indptr: Some((0..=n).collect()),
            positions: Some((16..16 + n).collect()),
            pages: Some((0..n).flat_map(|_| [6u32, 7]).collect()),
            page_indptr: Some((0..=n).map(|lane| lane * 2).collect()),
            w_slot: Some(vec![7; n as usize]),
            w_off: Some((0..n).collect()),
        };

        assert_eq!(
            canonical_hash_tokens(evidence, &request, false, 16),
            Some(CanonicalAppend { start: 16, tokens })
        );
    }

    #[test]
    fn in_band_skip_tokens_never_hash() {
        // -1 (0xFFFFFFFF) anywhere means this is not a plain append.
        let tokens = vec![1, u32::MAX, 3];
        let request = crate::driver::LaunchPlan {
            token_ids: tokens.clone(),
            position_ids: (0..3).collect(),
            qo_indptr: vec![0, 3],
            ..crate::driver::LaunchPlan::default()
        };
        let evidence = CanonicalFireEvidence {
            tokens,
            kv_len: vec![3],
            embed_indptr: Some(vec![0, 3]),
            positions: Some((0..3).collect()),
            pages: None,
            page_indptr: None,
            w_slot: None,
            w_off: None,
        };
        assert!(canonical_hash_tokens(evidence, &request, false, 16).is_none());
    }

    #[test]
    fn prefill_then_decode_grows_and_projects() {
        let mut store = KvStore::new(16, nonce());
        let ws = store.create_working_set();
        let page = 4u32;

        // Fresh prefill: 6 tokens -> 2 pages, both fresh writes.
        let (proj, (src, dst), _tr, txn) = prepare(
            &mut store,
            ws,
            0,
            &[1, 2, 3, 4, 5, 6],
            page,
            Some(&[1, 2, 3, 4, 5, 6]),
        )
        .unwrap();
        assert_eq!(proj.physical_page_ids.len(), 2);
        assert_eq!(proj.last_page_len, 2);
        assert!(src.is_empty() && dst.is_empty());
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.mapped_len(ws).unwrap(), 2);

        // Decode: one token into the private partial tail -> in-place write.
        let before = store.lookup(ws, 1).unwrap();
        let (proj, (src, _dst), _tr, txn) =
            prepare(&mut store, ws, 6, &[7], page, Some(&[7])).unwrap();
        assert_eq!(proj.physical_page_ids.len(), 2);
        assert_eq!(proj.last_page_len, 3);
        assert!(src.is_empty()); // private -> no CoW copies
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.lookup(ws, 1).unwrap(), before); // id stable in place
    }

    #[test]
    fn single_state_runahead_matches_synchronous_projection_translation_and_hashes() {
        let page = 4u32;
        let tokens: Vec<u32> = (1..=9).collect();
        let chunks = [5usize, 1, 3];

        let mut synchronous = KvStore::new(16, nonce());
        let sync_ws = synchronous.create_working_set();
        let mut sync_shapes = Vec::new();
        let mut offset = 0usize;
        for size in chunks {
            let chunk = &tokens[offset..offset + size];
            let (projection, _, translation, txn) = prepare(
                &mut synchronous,
                sync_ws,
                offset as u32,
                chunk,
                page,
                Some(chunk),
            )
            .unwrap();
            sync_shapes.push((projection, translation));
            finalize(&mut synchronous, txn, true).unwrap();
            offset += size;
        }

        let mut runahead = KvStore::new(16, nonce());
        let runahead_ws = runahead.create_working_set();
        let mut pending = Vec::new();
        let mut runahead_shapes = Vec::new();
        offset = 0;
        for size in chunks {
            let chunk = &tokens[offset..offset + size];
            let (projection, _, translation, txn) = prepare(
                &mut runahead,
                runahead_ws,
                offset as u32,
                chunk,
                page,
                Some(chunk),
            )
            .unwrap();
            runahead_shapes.push((projection, translation));
            pending.push(txn);
            offset += size;
        }

        assert_eq!(runahead.mapped_len(runahead_ws).unwrap(), 3);
        assert_eq!(
            runahead.committed_token_len(runahead_ws, page).unwrap(),
            tokens.len() as u64
        );
        assert_eq!(runahead_shapes, sync_shapes);

        for txn in pending {
            finalize(&mut runahead, txn, true).unwrap();
        }
        assert_eq!(
            runahead.chain_state(runahead_ws).unwrap(),
            synchronous.chain_state(sync_ws).unwrap()
        );
        for index in 0..tokens.len().div_ceil(page as usize) as u64 {
            assert_eq!(
                runahead.page_token_hashes(runahead_ws, index).unwrap(),
                synchronous.page_token_hashes(sync_ws, index).unwrap()
            );
            assert_eq!(
                runahead.page_hash_at(runahead_ws, index).unwrap(),
                synchronous.page_hash_at(sync_ws, index).unwrap()
            );
        }
    }

    #[test]
    fn single_state_reuses_cow_destination_and_preserves_hash_chain() {
        let mut store = KvStore::new(32, nonce());
        let page = 4u32;
        let parent = store.create_working_set();
        prefill(&mut store, parent, &[1, 2, 3, 4, 5, 6], &[6], page);
        let synchronous = store.fork(parent).unwrap();
        let runahead = store.fork(parent).unwrap();

        let (_, _, _, sync_first) =
            prepare(&mut store, synchronous, 6, &[7], page, Some(&[7])).unwrap();
        finalize(&mut store, sync_first, true).unwrap();
        let (_, _, _, sync_second) =
            prepare(&mut store, synchronous, 7, &[8], page, Some(&[8])).unwrap();
        finalize(&mut store, sync_second, true).unwrap();

        let (_, (first_copy_src, _), first_translation, first) =
            prepare(&mut store, runahead, 6, &[7], page, Some(&[7])).unwrap();
        assert!(!first_copy_src.is_empty());
        let (_, (second_copy_src, _), second_translation, second) =
            prepare(&mut store, runahead, 7, &[8], page, Some(&[8])).unwrap();
        assert!(
            second_copy_src.is_empty(),
            "the second fire writes the first fire's published private CoW destination"
        );
        assert_eq!(first_translation[1], second_translation[1]);
        finalize(&mut store, first, true).unwrap();
        finalize(&mut store, second, true).unwrap();

        assert_eq!(
            store.chain_state(runahead).unwrap(),
            store.chain_state(synchronous).unwrap()
        );
        assert_eq!(
            store.page_token_hashes(runahead, 1).unwrap(),
            store.page_token_hashes(synchronous, 1).unwrap()
        );
    }

    #[test]
    fn failed_runahead_keeps_fail_stop_mapping_until_release() {
        let mut store = KvStore::new(4, nonce());
        let ws = store.create_working_set();
        let (_, _, _, first) =
            prepare(&mut store, ws, 0, &[1, 2, 3, 4], 4, Some(&[1, 2, 3, 4])).unwrap();
        let (_, _, _, second) = prepare(&mut store, ws, 4, &[5], 4, Some(&[5])).unwrap();
        assert_eq!(store.available_pages(), 2);

        finalize(&mut store, first, false).unwrap();
        assert_eq!(
            store.available_pages(),
            2,
            "the downstream translation can still reference the predecessor allocation"
        );
        finalize(&mut store, second, false).unwrap();
        assert_eq!(store.available_pages(), 2);
        store.release_working_set(ws, store.current_epoch());
        store.retire_idle();
        assert_eq!(store.available_pages(), 4);
    }

    #[test]
    fn working_set_release_recycles_after_inflight_epoch_retires() {
        let mut store = KvStore::new(4, nonce());
        let ws = store.create_working_set();
        let (_, _, _, pending) =
            prepare(&mut store, ws, 0, &[1, 2, 3, 4], 4, Some(&[1, 2, 3, 4])).unwrap();
        store.release_working_set(ws, store.current_epoch());
        assert_eq!(store.available_pages(), 3);

        finalize(&mut store, pending, true).unwrap();
        assert_eq!(store.available_pages(), 4);
        assert!(store.mapped_len(ws).is_err());
    }

    #[test]
    fn forked_decode_cows_the_shared_tail() {
        let mut store = KvStore::new(16, nonce());
        let ws = store.create_working_set();
        let page = 4u32;
        let (_, _, _, txn) = prepare(
            &mut store,
            ws,
            0,
            &[1, 2, 3, 4, 5, 6],
            page,
            Some(&[1, 2, 3, 4, 5, 6]),
        )
        .unwrap();
        finalize(&mut store, txn, true).unwrap();

        let forked = store.fork(ws).unwrap();
        let shared_tail = store.lookup(forked, 1).unwrap();
        let (proj, (src, dst), _tr, txn) =
            prepare(&mut store, forked, 6, &[7], page, Some(&[7])).unwrap();
        assert_eq!(src, vec![shared_tail.0]); // preserved cells copied
        assert_eq!(dst.len(), 1);
        assert_ne!(proj.physical_page_ids[1], shared_tail.0);
        finalize(&mut store, txn, true).unwrap();
        // The original keeps its tail.
        assert_eq!(store.lookup(ws, 1).unwrap(), shared_tail);
        assert_ne!(store.lookup(forked, 1).unwrap(), shared_tail);
    }

    #[test]
    fn failed_fire_leaves_fail_stop_mapping_published() {
        let mut store = KvStore::new(4, nonce());
        let ws = store.create_working_set();
        let (_, _, _, txn) = prepare(&mut store, ws, 0, &[1, 2], 4, None).unwrap();
        finalize(&mut store, txn, false).unwrap();
        assert_eq!(store.mapped_len(ws).unwrap(), 1);
        assert_eq!(store.available_pages(), 3);
    }

    #[test]
    fn declaration_realization_cows_only_a_shared_mapped_tail() {
        let mut store = KvStore::new(8, nonce());
        let parent = store.create_working_set();
        prefill(&mut store, parent, &(1..=8).collect::<Vec<_>>(), &[8], 4);
        let parent_tail = store.lookup(parent, 1).unwrap();

        let child = store.fork(parent).unwrap();
        let ((copy_src, copy_dst), txn) = realize_declaration(&mut store, child, 1..2).unwrap();
        assert_eq!(copy_src, vec![parent_tail.0]);
        assert_eq!(copy_dst.len(), 1);
        assert_ne!(store.lookup(child, 1).unwrap(), parent_tail);
        assert_eq!(store.lookup(parent, 1).unwrap(), parent_tail);
        assert!(store.page_token_hashes(child, 1).unwrap().is_empty());
        finalize(&mut store, txn.unwrap(), true).unwrap();
    }

    /// Canonical prefill of `tokens` onto `ws`, chunked as `fires` splits.
    fn prefill(store: &mut KvStore, ws: WorkingSetId, tokens: &[u32], fires: &[usize], page: u32) {
        let mut done = 0usize;
        for &n in fires {
            let chunk = &tokens[done..done + n];
            let (_, _, _, txn) = prepare(store, ws, done as u32, chunk, page, Some(chunk)).unwrap();
            finalize(store, txn, true).unwrap();
            done += n;
        }
    }

    #[test]
    fn canonical_hashes_are_fire_chunking_independent() {
        let mut store = KvStore::new(16, nonce());
        let page = 4u32;
        let tokens: Vec<u32> = (100..108).collect(); // 8 tokens = 2 full pages

        let a = store.create_working_set();
        prefill(&mut store, a, &tokens, &[8], page);
        let b = store.create_working_set();
        prefill(&mut store, b, &tokens, &[5, 3], page); // partial page finished in place

        for i in 0..2u64 {
            assert_eq!(
                store.page_token_hashes(a, i).unwrap(),
                store.page_token_hashes(b, i).unwrap(),
                "slot hashes differ at page {i}"
            );
            let (ha, hb) = (
                store.page_hash_at(a, i).unwrap(),
                store.page_hash_at(b, i).unwrap(),
            );
            assert!(ha.is_some(), "full canonical page has a page hash");
            assert_eq!(ha, hb, "page hashes differ at page {i}");
        }

        // CAS: both boundary chain values are indexed and validate live.
        for i in 0..2u64 {
            let key = store.page_token_hashes(a, i).unwrap()[3].unwrap();
            assert!(store.lookup_cached_page(&key).is_some());
        }
    }

    #[test]
    fn opaque_fires_never_produce_matchable_identity() {
        let mut store = KvStore::new(16, nonce());
        let page = 4u32;
        let tokens = [1u32, 2, 3, 4];

        let a = store.create_working_set();
        let (_, _, _, txn) = prepare(&mut store, a, 0, &tokens, page, None).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let b = store.create_working_set();
        let (_, _, _, txn) = prepare(&mut store, b, 0, &tokens, page, None).unwrap();
        finalize(&mut store, txn, true).unwrap();

        // Same tokens, but the fires were not canonical: identities differ
        // and no page hash marks them for the CAS index.
        assert_ne!(
            store.page_token_hashes(a, 0).unwrap(),
            store.page_token_hashes(b, 0).unwrap()
        );
        assert_eq!(store.page_hash_at(a, 0).unwrap(), None);
    }

    #[test]
    fn fork_continuations_hash_identically() {
        let mut store = KvStore::new(16, nonce());
        let page = 4u32;
        let a = store.create_working_set();
        prefill(&mut store, a, &[1, 2, 3, 4, 5, 6], &[6], page);
        let b = store.fork(a).unwrap();

        // The same next token on both branches (one CoW, one shared-blocked
        // CoW as well) must produce the same slot identity.
        let (_, _, _, txn) = prepare(&mut store, b, 6, &[7], page, Some(&[7])).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let (_, _, _, txn) = prepare(&mut store, a, 6, &[7], page, Some(&[7])).unwrap();
        finalize(&mut store, txn, true).unwrap();

        assert_eq!(
            store.page_token_hashes(a, 1).unwrap(),
            store.page_token_hashes(b, 1).unwrap()
        );
        // The shared full prefix page kept one identity.
        assert_eq!(
            store.page_hash_at(a, 0).unwrap(),
            store.page_hash_at(b, 0).unwrap()
        );
    }

    #[test]
    fn translation_overlays_prepared_targets_on_the_committed_mapping() {
        let mut store = KvStore::new(16, nonce());
        let ws = store.create_working_set();
        let page = 4u32;

        // Prefill: both entries are this fire's fresh targets.
        let (proj, _, tr, txn) =
            prepare(&mut store, ws, 0, &[1, 2, 3, 4, 5, 6], page, None).unwrap();
        assert_eq!(tr, proj.physical_page_ids);
        finalize(&mut store, txn, true).unwrap();

        // Forked decode: entry 0 = shared committed page, entry 1 = the CoW
        // destination of THIS fire (not the shared source).
        let forked = store.fork(ws).unwrap();
        let shared_head = store.lookup(forked, 0).unwrap().0;
        let shared_tail = store.lookup(forked, 1).unwrap().0;
        let (_, _, tr, txn) = prepare(&mut store, forked, 6, &[7], page, None).unwrap();
        assert_eq!(tr[0], shared_head);
        assert_ne!(tr[1], shared_tail);
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.lookup(forked, 1).unwrap().0, tr[1]);
    }

    #[test]
    fn prefix_match_grafts_shared_pages_and_continues_the_chain() {
        let mut store = KvStore::new(32, nonce());
        let page = 4u32;
        let tokens: Vec<u32> = (300..312).collect(); // 12 tokens

        // Producer: canonical 8-token prefill (2 full pages, CAS-indexed).
        let a = store.create_working_set();
        prefill(&mut store, a, &tokens[..8], &[8], page);

        // Fresh consumer prefilling all 12: matches the 2-page prefix.
        let b = store.create_working_set();
        let matched = match_prefix(&mut store, b, &tokens, page)
            .unwrap()
            .expect("prefix hit");
        assert_eq!(matched, 2);
        // Structurally shared: b's visible pages ARE a's physical pages.
        assert_eq!(store.lookup(b, 0).unwrap(), store.lookup(a, 0).unwrap());
        assert_eq!(store.lookup(b, 1).unwrap(), store.lookup(a, 1).unwrap());

        // Continue as a committed-jump fire: hashes must equal a straight
        // 12-token prefill's (the dedup property end-to-end).
        let (_, _, _, txn) =
            prepare(&mut store, b, 8, &tokens[8..], page, Some(&tokens[8..])).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let c = store.create_working_set();
        prefill(&mut store, c, &tokens, &[12], page);
        for i in 0..3u64 {
            assert_eq!(
                store.page_token_hashes(b, i).unwrap(),
                store.page_token_hashes(c, i).unwrap(),
                "grafted continuation diverged at page {i}"
            );
        }

        // The match never swallows the whole prompt: with exactly 2 pages of
        // tokens, only 1 page may match (the readout row must compute).
        let d = store.create_working_set();
        let matched = match_prefix(&mut store, d, &tokens[..8], page)
            .unwrap()
            .expect("capped prefix hit");
        assert_eq!(matched, 1);
    }

    #[test]
    fn front_surgery_breaks_the_chain_but_tail_surgery_continues_it() {
        let mut store = KvStore::new(32, nonce());
        let page = 4u32;
        let tokens: Vec<u32> = (200..212).collect(); // 3 full pages

        // Reference: 8 tokens then append X.
        let a = store.create_working_set();
        prefill(&mut store, a, &tokens[..8], &[8], page);
        let (_, _, _, txn) = prepare(&mut store, a, 8, &[999], page, Some(&[999])).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let a_x = store.page_token_hashes(a, 2).unwrap()[0].unwrap();

        // Tail discard: 12 tokens, drop page 2 -> visible content == a's
        // first 8 tokens. Appending X must hash EXACTLY like a's append.
        let b = store.create_working_set();
        prefill(&mut store, b, &tokens, &[12], page);
        let epoch = store.current_epoch();
        store.discard(b, &[2..3], epoch).unwrap();
        let (_, _, _, txn) = prepare(&mut store, b, 8, &[999], page, Some(&[999])).unwrap();
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.page_token_hashes(b, 2).unwrap()[0].unwrap(), a_x);

        // Front discard: 12 tokens, drop page 0 -> 8 visible tokens but a
        // DIFFERENT context. Appending X must NOT impersonate a's append.
        let c = store.create_working_set();
        prefill(&mut store, c, &tokens, &[12], page);
        let epoch = store.current_epoch();
        store.discard(c, &[0..1], epoch).unwrap();
        let (_, _, _, txn) = prepare(&mut store, c, 8, &[999], page, Some(&[999])).unwrap();
        finalize(&mut store, txn, true).unwrap();
        assert_ne!(store.page_token_hashes(c, 2).unwrap()[0].unwrap(), a_x);
    }
}
