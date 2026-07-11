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

use crate::pipeline::channel::BoundCells;
use crate::pipeline::instance::ChannelSeed;
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
    prepared: KvPreparedWrite,
    /// Per-target committed metadata computed at prepare (the token values
    /// and pre-fire chain state exist only then); handed to `KvStore::commit`
    /// at finalize.
    commits: Vec<PageCommit>,
    /// The working set's committed token length AFTER this fire commits
    /// (`committed_tokens + new_tokens.len()`). The caller stores it on the
    /// pass (the growing cursor) and passes it back next fire.
    pub committed_tokens_after: u32,
}

/// The fire's WorkingSet page translation (kv_refact.md flattened-table
/// model): entry `i` = the physical page backing WS-relative index `i`,
/// i.e. the committed flat table overlaid with this fire's prepared write
/// targets. Ships with the launch so the driver can map channel-resolved
/// `Pages`/`WSlot` references; guests only ever hold relative indexes.
fn build_translation(
    store: &mut KvStore,
    prepared: &KvPreparedWrite,
    ws: WorkingSetId,
) -> Result<Vec<u32>, KvStoreError> {
    let mut table: Vec<u32> = {
        let (_, flat) = store.flat_table(ws)?;
        flat.iter().map(|p| p.0).collect()
    };
    let max_target = prepared
        .targets()
        .iter()
        .map(|t| t.index() + 1)
        .max()
        .unwrap_or(0);
    if (table.len() as u64) < max_target {
        table.resize(max_target as usize, 0);
    }
    for t in prepared.targets() {
        table[t.index() as usize] = t.dst().0;
    }
    Ok(table)
}

/// Build the per-target [`PageCommit`]s for a fire appending `n_new` tokens
/// at `committed_tokens`. `hash_tokens = Some(values)` on a canonical fire
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
    committed_tokens: u32,
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
                committed_tokens + j,
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
        // [committed, committed + n_new) landing on page `page`.
        let mut wrote = false;
        for (j, h) in slot_hashes.iter().enumerate() {
            let tok = committed_tokens as u64 + j as u64;
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
/// (currently holding `committed_tokens` committed tokens).
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
    committed_tokens: u32,
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
    let total = committed_tokens + n_new;
    let needed_pages = total.div_ceil(page_size) as u64;

    // Grow the logical address space so every write slot exists. Purely
    // logical: physical pages are allocated by prepare_write below.
    let page_len = store.page_len(ws)?;
    if page_len < needed_pages {
        store.reserve(ws, needed_pages - page_len)?;
    }

    // Prior context: pages [0, valid_pages) for the committed tokens, from
    // the flattened table (write targets override their slots below).
    let valid_pages = (committed_tokens.div_ceil(page_size)) as usize;
    let context_pages: Vec<u32> = {
        let (_, flat) = store.flat_table(ws)?;
        flat.iter().take(valid_pages).map(|p| p.0).collect()
    };
    if context_pages.len() < valid_pages {
        return Err(KvError::Fatal(format!(
            "prepare: committed {committed_tokens} tokens but only {} mapped pages",
            context_pages.len()
        )));
    }

    // Classify + allocate the write slots [output_start, needed_pages).
    // `KvStoreError::OutOfPages` stays typed through here — the caller
    // routes it into the contention ladder and retries.
    let output_start = (committed_tokens / page_size) as u64;
    let write_indexes: Vec<u64> = (output_start..needed_pages).collect();
    let prepared = store.prepare_write(ws, &write_indexes)?;

    // Driver geometry: every prepared target is a written slot (the CoW
    // rebase never reaches below the first written committed page).
    let offset = committed_tokens % page_size;
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

    let proj = project_kv(&context_pages, committed_tokens, &writes, page_size).map_err(|e| {
        KvError::Fatal(format!(
            "{e:?} (committed={committed_tokens}, new={n_new}, targets={:?})",
            prepared
                .targets()
                .iter()
                .map(|t| t.index())
                .collect::<Vec<_>>()
        ))
    })?;

    let commits = build_commits(
        store,
        &prepared,
        ws,
        committed_tokens,
        n_new,
        page_size,
        hash_tokens,
    )?;
    let translation = build_translation(store, &prepared, ws)?;

    Ok((
        proj,
        (copy_src, copy_dst),
        translation,
        KvTxn {
            prepared,
            commits,
            committed_tokens_after: total,
        },
    ))
}

/// Prepare an explicit-KV (device-geometry) fire: physical pages for
/// `write_indexes` with no host projection — the driver resolves the geometry
/// itself and the inferlet owns the token bookkeeping. Returns the
/// `(index, physical id)` pairs for the granted slots, the CoW copy plan, and
/// the held txn (its `committed_tokens_after` is unused on this path).
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
    let translation = build_translation(store, &prepared, ws)?;
    // Device-geometry fires are non-canonical by construction, and the
    // device owns the token bookkeeping — commit with no hash metadata;
    // `KvStore::commit` poisons the chain state with an opaque draw.
    let commits = prepared
        .targets()
        .iter()
        .map(|_| PageCommit {
            token_hashes: Vec::new(),
            page_hash: None,
        })
        .collect();
    Ok((
        pages,
        copies,
        translation,
        KvTxn {
            prepared,
            commits,
            committed_tokens_after: 0,
        },
    ))
}

/// Abandon a fire's prepared KV write (e.g. the guest dropped the working set
/// while the fire was in flight): the snapshot pin kept the captured path
/// alive, so abort releases it and the pending slots safely.
pub fn abandon(store: &mut KvStore, txn: KvTxn) {
    let seq = txn.prepared.seq();
    let epoch = store.current_epoch();
    store.abort(txn.prepared, epoch);
    store.retire_through(seq);
}

/// Finalize a PTIR fire's KV write after `submit_async` resolves. `success`
/// publishes the mapping (pages persist for the next fire); otherwise the
/// pending slots release and the committed mapping is untouched. Fires retire
/// in FIFO stream order, so this fire's sequence retires every recycle tagged
/// at or before it.
pub fn finalize(store: &mut KvStore, txn: KvTxn, success: bool) -> Result<(), String> {
    let KvTxn {
        prepared, commits, ..
    } = txn;
    let seq = prepared.seq();
    let epoch = store.current_epoch();
    if success {
        store
            .commit(prepared, &commits, epoch)
            .map_err(|e| e.to_string())?;
    } else {
        store.abort(prepared, epoch);
    }
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
/// an attention mask or explicit positions (they change hidden states, hence
/// KV at layers > 0), device geometry (inferlet-managed layout), per-layer
/// stage programs (they can rewrite projections), extern channels, or
/// multi-lane batching. Prologue/epilogue programs only shape sampling —
/// grammar, watermarking, and sampler passes all stay canonical. A `KvLen`
/// port must exist so the fire-time gate can verify the pass attends the
/// FULL context (a shorter span changes upper-layer KV).
pub fn canonical_kv_shape(container: &pie_ptir::container::TraceContainer) -> bool {
    use pie_ptir::container::PortSource;
    use pie_ptir::registry::{Port, Stage};

    if !container.externs.is_empty() {
        return false;
    }
    let mut has_kv_len = false;
    for binding in &container.ports {
        match binding.port {
            Port::AttnMask
            | Port::Positions
            | Port::Pages
            | Port::PageIndptr
            | Port::WSlot
            | Port::WOff => return false,
            Port::KvLen => has_kv_len = true,
            Port::EmbedIndptr => {
                // Single lane only: a trace-const two-entry CSR from 0.
                match &binding.source {
                    PortSource::Const { data, .. } => {
                        if data.len() != 8 || data[0..4] != [0u8; 4] {
                            return false;
                        }
                    }
                    PortSource::Channel(_) => return false,
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

/// Reinterpret a little-endian byte payload as `u32`s.
fn decode_le_u32s(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// The host-known u32 payload the channel bound at `port` consumes on THIS
/// fire: a trace constant, the staged Writer put shipping with this fire, or
/// (first fire only) the channel's seed. `None` = device-carried — the host
/// cannot see the value, so the fire cannot hash canonically. Takes the
/// pass's plain pipeline-owned state (`container`/`cells`/`seeds`/
/// `fired_once`) rather than the WIT-resource `ForwardPass` itself, so this
/// stays a `pipeline::fire` function (no upward `inferlet::host` import).
pub fn host_known_port_u32s(
    container: &pie_ptir::container::TraceContainer,
    cells: &BoundCells,
    seeds: &[ChannelSeed],
    fired_once: bool,
    port: pie_ptir::registry::Port,
) -> Option<Vec<u32>> {
    use pie_ptir::container::PortSource;

    let binding = container.ports.iter().find(|b| b.port == port)?;
    match &binding.source {
        PortSource::Const { data, .. } => decode_le_u32s(data),
        PortSource::Channel(c) => {
            let dense = *c as usize;
            if let Some(bytes) = crate::pipeline::channel::staged_put_bytes(cells.get(dense)?) {
                return decode_le_u32s(&bytes);
            }
            if !fired_once && container.channels.get(dense).is_some_and(|d| d.seeded) {
                let seed = seeds.iter().find(|s| s.channel as usize == dense)?;
                return decode_le_u32s(&seed.data);
            }
            None
        }
    }
}

/// Fire-time half of the canonical-KV gate: the host-verified token values
/// this fire embeds plus the kv-len the pass claims to attend. `None` unless
/// the bind-time shape passed ([`canonical_kv_shape`], captured as
/// `canonical_kv` at bind) and the embed value is host-known. The caller
/// still verifies the token count against the fire geometry and
/// `kv_len == committed + new` (full-context attention) before hashing.
pub fn canonical_fire_evidence(
    canonical_kv: bool,
    container: &pie_ptir::container::TraceContainer,
    cells: &BoundCells,
    seeds: &[ChannelSeed],
    fired_once: bool,
) -> Option<(Vec<u32>, Option<u32>)> {
    use pie_ptir::registry::Port;

    if !canonical_kv {
        return None;
    }
    let tokens = host_known_port_u32s(container, cells, seeds, fired_once, Port::EmbedTokens)?;
    let kv_len = host_known_port_u32s(container, cells, seeds, fired_once, Port::KvLen)
        .and_then(|v| v.first().copied());
    Some((tokens, kv_len))
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

    /// A minimal canonical decode container: embed tokens + kv-len +
    /// epilogue. Channels: 0 tok (device-loop), 1 klen.
    fn plain_decode_container() -> pie_ptir::container::TraceContainer {
        pie_ptir::container::TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::vector(1), DType::I32, HostRole::None),
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
        assert_eq!(txn.committed_tokens_after, 6);
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
    fn failed_fire_leaves_committed_state_untouched() {
        let mut store = KvStore::new(4, nonce());
        let ws = store.create_working_set();
        let (_, _, _, txn) = prepare(&mut store, ws, 0, &[1, 2], 4, None).unwrap();
        finalize(&mut store, txn, false).unwrap();
        assert_eq!(store.mapped_len(ws).unwrap(), 0);
        assert_eq!(store.available_pages(), 4); // retired via FIFO seq
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
    fn released_paths_stay_matchable_until_pressure_reclaims_them() {
        let mut store = KvStore::new(8, nonce());
        let page = 4u32;
        let tokens: Vec<u32> = (400..408).collect();

        let a = store.create_working_set();
        prefill(&mut store, a, &tokens, &[8], page);
        let epoch = store.current_epoch();
        store.release_working_set_cached(a, epoch, 8);
        store.retire_idle();
        // Retained: the pages did NOT free.
        assert_eq!(store.available_pages(), 6);

        // A newcomer still matches the retained prefix.
        let b = store.create_working_set();
        let matched = match_prefix(&mut store, b, &tokens, page).unwrap();
        assert_eq!(matched, Some(1)); // 8 tokens -> capped at 1 full page
        let epoch = store.current_epoch();
        store.release_working_set(b, epoch);
        store.retire_idle();

        // Rung 1 reclaims the retained lease; nothing matches afterwards.
        let epoch = store.current_epoch();
        assert_eq!(store.drop_unused_cache_leases(epoch), 2);
        store.retire_idle();
        assert_eq!(store.available_pages(), 8);
        let c = store.create_working_set();
        assert_eq!(match_prefix(&mut store, c, &tokens, page).unwrap(), None);
    }

    #[test]
    fn retention_cap_evicts_the_oldest_root() {
        let mut store = KvStore::new(16, nonce());
        let page = 4u32;
        let t1: Vec<u32> = (500..504).collect();
        let t2: Vec<u32> = (600..604).collect();

        for t in [&t1, &t2] {
            let ws = store.create_working_set();
            prefill(&mut store, ws, t, &[4], page);
            let epoch = store.current_epoch();
            store.release_working_set_cached(ws, epoch, 1); // cap 1
            store.retire_idle();
        }
        // t1's root was evicted by t2's retention; only t2 lingers.
        assert_eq!(store.available_pages(), 15);
        let probe = store.create_working_set();
        assert_eq!(
            match_prefix(&mut store, probe, &[t1.as_slice(), &[9]].concat(), page).unwrap(),
            None
        );
        let probe2 = store.create_working_set();
        assert_eq!(
            match_prefix(&mut store, probe2, &[t2.as_slice(), &[9]].concat(), page).unwrap(),
            Some(1)
        );
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
