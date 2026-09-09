#![allow(dead_code)]

use crate::store::kv::hash::{self, Hash256};
use crate::store::kv::page_table::{PhysicalKvPageId, WorkingSetId};
use crate::store::kv::project::{KvProjection, KvWrite, project_kv};
use crate::store::kv::write::{KvPreparedWrite, PageCommit, PreparedTarget};
use crate::store::kv::{KvStore, KvStoreError};

#[derive(Debug)]
pub enum KvError {
    OutOfPages { requested: usize, available: usize },
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

fn build_translation(
    store: &mut KvStore,
    ws: WorkingSetId,
) -> Result<(u64, Vec<u32>), KvStoreError> {
    let (version, table) = store.flat_table(ws)?;
    Ok((version, table.iter().map(|page| page.0).collect()))
}

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

        let mut wrote = false;
        for (j, h) in slot_hashes.iter().enumerate() {
            let tok = append_start as u64 + j as u64;
            if tok / page_size as u64 == page {
                hashes[(tok % page_size as u64) as usize] = Some(*h);
                wrote = true;
            }
        }

        let page_hash = if !wrote {
            existing_page_hash
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

fn declaration_overlap(
    store: &mut KvStore,
    ws: WorkingSetId,
    writable: std::ops::Range<u64>,
) -> Result<Option<(u64, Vec<u64>, bool)>, KvError> {
    let mapped = store.mapped_len(ws)?;
    let start = writable.start.min(mapped);
    let end = writable.end.min(mapped);
    if start >= end {
        return Ok(None);
    }
    let indexes: Vec<u64> = (start..end).collect();
    let shared = indexes
        .iter()
        .copied()
        .map(|index| store.privately_writable(ws, index))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .any(|private| !private);
    Ok(Some((start, indexes, shared)))
}

pub type PageCopies = (Vec<u32>, Vec<u32>);

pub type RealizedDeclaration = (PageCopies, Option<KvTxn>);

pub type PreparedAppend = (KvProjection, PageCopies, Vec<u32>, KvTxn);

pub type PreparedExplicit = (Vec<(u64, u32)>, PageCopies, Vec<u32>, KvTxn);

#[cfg(test)]
pub fn realize_declaration(
    store: &mut KvStore,
    ws: WorkingSetId,
    writable: std::ops::Range<u64>,
) -> Result<RealizedDeclaration, KvError> {
    realize_declaration_impl(store, ws, writable, None)
}

pub fn realize_ahead_range(
    store: &KvStore,
    ws: WorkingSetId,
    writable: &std::ops::Range<u64>,
    ahead: u64,
) -> Result<std::ops::Range<u64>, KvError> {
    if ahead == 0 {
        return Ok(writable.clone());
    }
    let capacity = store.page_len(ws)?;
    let end = writable
        .end
        .saturating_add(ahead)
        .min(capacity)
        .max(writable.end);
    Ok(writable.start..end)
}

pub fn realize_declaration_demand(
    store: &mut KvStore,
    ws: WorkingSetId,
    writable: std::ops::Range<u64>,
) -> Result<usize, KvError> {
    match declaration_overlap(store, ws, writable)? {
        Some((_, indexes, true)) => Ok(store.write_demand(ws, &indexes)?),
        _ => Ok(0),
    }
}

pub fn realize_declaration_reserved(
    store: &mut KvStore,
    ws: WorkingSetId,
    writable: std::ops::Range<u64>,
    granted: &mut Vec<PhysicalKvPageId>,
) -> Result<RealizedDeclaration, KvError> {
    realize_declaration_impl(store, ws, writable, Some(granted))
}

fn realize_declaration_impl(
    store: &mut KvStore,
    ws: WorkingSetId,
    writable: std::ops::Range<u64>,
    granted: Option<&mut Vec<PhysicalKvPageId>>,
) -> Result<RealizedDeclaration, KvError> {
    let Some((start, indexes, shared)) = declaration_overlap(store, ws, writable)? else {
        return Ok(((Vec::new(), Vec::new()), None));
    };
    if !shared {
        store.opacify_suffix(ws, start)?;
        return Ok(((Vec::new(), Vec::new()), None));
    }

    let prepared = match granted {
        Some(granted) => store.prepare_write_reserved(ws, &indexes, granted)?,
        None => store.prepare_write(ws, &indexes)?,
    };
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

pub fn match_prefix(
    store: &mut KvStore,
    ws: WorkingSetId,
    tokens: &[u32],
    page_size: u32,
) -> Result<Option<u64>, KvError> {
    if store.mapped_len(ws)? != 0 || store.chain_state(ws)?.is_some() {
        return Ok(None);
    }
    let ps = page_size as usize;
    let max_pages = tokens.len().saturating_sub(1) / ps;
    if max_pages == 0 {
        return Ok(None);
    }
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

pub fn prepare(
    store: &mut KvStore,
    ws: WorkingSetId,
    append_start: u32,
    new_tokens: &[u32],
    page_size: u32,
    hash_tokens: Option<&[u32]>,
) -> Result<PreparedAppend, KvError> {
    debug_assert!(hash_tokens.is_none_or(|t| t.len() == new_tokens.len()));
    let n_new = new_tokens.len() as u32;
    if n_new == 0 {
        return Err(KvError::Fatal(
            "prepare: new_tokens must be non-empty".to_string(),
        ));
    }

    let total = append_start + n_new;
    let needed_pages = total.div_ceil(page_size) as u64;

    let page_len = store.page_len(ws)?;
    if page_len < needed_pages {
        store.reserve(ws, needed_pages - page_len)?;
    }

    let valid_pages = (append_start.div_ceil(page_size)) as usize;
    let context_pages: Vec<u32> = {
        store
            .flat_table(ws)?
            .1
            .iter()
            .copied()
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

    let output_start = (append_start / page_size) as u64;
    let write_indexes: Vec<u64> = (output_start..needed_pages).collect();
    let prepared = store.prepare_write(ws, &write_indexes)?;

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
            store.settle(seq, cas_intents, false);
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

pub fn prepare_explicit_demand(
    store: &mut KvStore,
    ws: WorkingSetId,
    write_indexes: &[u64],
) -> Result<usize, KvError> {
    Ok(store.write_demand(ws, write_indexes)?)
}

pub fn prepare_explicit_reserved(
    store: &mut KvStore,
    ws: WorkingSetId,
    write_indexes: &[u64],
    granted: &mut Vec<PhysicalKvPageId>,
) -> Result<PreparedExplicit, KvError> {
    let prepared = store.prepare_write_reserved(ws, write_indexes, granted)?;
    let pages: Vec<(u64, u32)> = prepared
        .targets()
        .iter()
        .map(|target| (target.index(), target.dst().0))
        .collect();
    let copies = prepared
        .copy_plan()
        .map(|(src, dst)| (src.0, dst.0))
        .unzip();
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
            store.settle(seq, cas_intents, false);
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

pub fn abandon(store: &mut KvStore, txn: KvTxn) {
    let KvTxn {
        seq, cas_intents, ..
    } = txn;
    store.settle(seq, cas_intents, false);
}

pub fn finalize(store: &mut KvStore, txn: KvTxn, success: bool) -> Result<(), String> {
    let KvTxn {
        seq, cas_intents, ..
    } = txn;
    store.settle(seq, cas_intents, success);
    Ok(())
}

pub fn canonical_kv_shape(container: &eta_ir::container::TraceContainer) -> bool {
    use eta_ir::container::PortSource;
    use eta_ir::registry::{Port, Stage};

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
    kv_len: Vec<u32>,
    embed_indptr: Option<Vec<u32>>,
    positions: Option<Vec<u32>>,
    pages: Option<Vec<u32>>,
    page_indptr: Option<Vec<u32>>,
    w_slot: Option<Vec<u32>>,
    w_off: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalAppend {
    pub start: u32,
    pub tokens: Vec<u32>,
}

pub fn canonical_hash_tokens(
    evidence: CanonicalFireEvidence,
    request: &crate::engine::FireRequest,
    device_resolved: bool,
    page_size: u32,
) -> Option<CanonicalAppend> {
    let n = evidence.tokens.len();
    if n == 0 || page_size == 0 || evidence.tokens.contains(&u32::MAX) {
        return None;
    }

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

    if let Some(positions) = &evidence.positions
        && (positions.len() != n
            || positions
                .iter()
                .enumerate()
                .any(|(i, &p)| p != start + i as u32))
    {
        return None;
    }

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

    if !device_resolved {
        let submitted: Vec<u32> = request
            .lanes
            .iter()
            .flat_map(|lane| lane.tokens.iter().copied())
            .collect();
        if evidence.tokens != submitted {
            return None;
        }
        if let Some(positions) = &evidence.positions {
            let mut at = 0usize;
            for lane in &request.lanes {
                let rows = lane.tokens.len();
                let stated: Vec<u32> = if lane.positions.is_empty() {
                    (0..rows as u32).map(|row| lane.kv.held + row).collect()
                } else {
                    lane.positions.clone()
                };
                if positions.get(at..at + rows) != Some(stated.as_slice()) {
                    return None;
                }
                at += rows;
            }
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

    fn one_lane(tokens: &[u32], held: u32) -> crate::engine::FireRequest {
        crate::engine::FireRequest::one(crate::engine::fire::lane_of(
            0,
            tokens.to_vec(),
            held,
            Vec::new(),
        ))
    }

    fn per_token_lanes(tokens: &[u32], held: u32) -> crate::engine::FireRequest {
        crate::engine::FireRequest {
            lanes: tokens
                .iter()
                .enumerate()
                .map(|(at, &token)| {
                    crate::engine::fire::lane_of(0, vec![token], held + at as u32, Vec::new())
                })
                .collect(),
            ..crate::engine::FireRequest::default()
        }
    }

    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, PortBinding, StageProgram};
    use eta_ir::registry::{Port, Stage};
    use eta_ir::types::{Dtype, Shape};

    fn ch(shape: Shape, dtype: Dtype, role: HostRole) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded: false,
        }
    }

    fn plain_decode_container() -> eta_ir::container::TraceContainer {
        eta_ir::container::TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::vector(1), Dtype::I32, HostRole::None),
                ch(Shape::vector(1), Dtype::U32, HostRole::None),
                ch(Shape::vector(4), Dtype::U32, HostRole::None),
                ch(Shape::vector(2), Dtype::U32, HostRole::None),
                ch(Shape::vector(1), Dtype::U32, HostRole::None),
                ch(Shape::vector(1), Dtype::U32, HostRole::None),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: eta_ir::container::PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::KvLen,
                    source: eta_ir::container::PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::Pages,
                    source: eta_ir::container::PortSource::Channel(2),
                },
                PortBinding {
                    port: Port::PageIndptr,
                    source: eta_ir::container::PortSource::Channel(3),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: eta_ir::container::PortSource::Channel(4),
                },
                PortBinding {
                    port: Port::WOff,
                    source: eta_ir::container::PortSource::Channel(5),
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
    fn kv_every_case() {
        canonical_shape_accepts_the_plain_decode();
        canonical_shape_rejects_kv_perturbing_passes();
        canonical_explicit_prefill_requires_contiguous_resolved_writes();
        prefill_then_decode_grows_and_projects();
        failed_runahead_keeps_fail_stop_mapping_until_release();
        forked_decode_cows_the_shared_tail();
        declaration_realization_cows_only_a_shared_mapped_tail();
        translation_overlays_prepared_targets_on_the_committed_mapping();
    }

    fn canonical_shape_accepts_the_plain_decode() {
        assert!(canonical_kv_shape(&plain_decode_container()));
    }

    fn canonical_shape_rejects_kv_perturbing_passes() {
        let mut c = plain_decode_container();
        c.ports.push(PortBinding {
            port: Port::AttnMask,
            source: eta_ir::container::PortSource::Channel(0),
        });
        assert!(!canonical_kv_shape(&c));

        let mut c = plain_decode_container();
        c.stages.push(StageProgram {
            stage: Stage::OnAttn,
            ops: vec![],
        });
        assert!(!canonical_kv_shape(&c));

        let mut c = plain_decode_container();
        c.ports.retain(|p| p.port != Port::KvLen);
        assert!(!canonical_kv_shape(&c));

        let devgeo = eta_ir::container::TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::matrix(2, 3), Dtype::U32, HostRole::None),
                ch(Shape::vector(2), Dtype::U32, HostRole::None),
                ch(Shape::vector(2), Dtype::U32, HostRole::None),
            ],
            ports: vec![
                PortBinding {
                    port: Port::Pages,
                    source: eta_ir::container::PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: eta_ir::container::PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::WOff,
                    source: eta_ir::container::PortSource::Channel(2),
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

    fn canonical_explicit_prefill_requires_contiguous_resolved_writes() {
        let tokens = (1..=17).collect::<Vec<_>>();
        let mut request = one_lane(&tokens, 0);
        let evidence = explicit_single_lane_evidence(&tokens, 0);
        assert_eq!(
            canonical_hash_tokens(evidence, &request, false, 16),
            Some(CanonicalAppend {
                start: 0,
                tokens: tokens.clone(),
            })
        );

        request.lanes[0].positions = (0..17).map(|at| if at == 16 { 0 } else { at }).collect();
        let invalid = explicit_single_lane_evidence(&tokens, 0);
        assert!(canonical_hash_tokens(invalid, &request, false, 16).is_none());
    }

    fn prefill_then_decode_grows_and_projects() {
        let mut store = KvStore::new(16, nonce());
        let ws = store.create_working_set();
        let page = 4u32;

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

        let before = store.lookup(ws, 1).unwrap();
        let (proj, (src, _dst), _tr, txn) =
            prepare(&mut store, ws, 6, &[7], page, Some(&[7])).unwrap();
        assert_eq!(proj.physical_page_ids.len(), 2);
        assert_eq!(proj.last_page_len, 3);
        assert!(src.is_empty());
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.lookup(ws, 1).unwrap(), before);
    }

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

        let forked = store.fork(ws, Default::default()).unwrap();
        let shared_tail = store.lookup(forked, 1).unwrap();
        let (proj, (src, dst), _tr, txn) =
            prepare(&mut store, forked, 6, &[7], page, Some(&[7])).unwrap();
        assert_eq!(src, vec![shared_tail.0]);
        assert_eq!(dst.len(), 1);
        assert_ne!(proj.physical_page_ids[1], shared_tail.0);
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.lookup(ws, 1).unwrap(), shared_tail);
        assert_ne!(store.lookup(forked, 1).unwrap(), shared_tail);
    }

    fn declaration_realization_cows_only_a_shared_mapped_tail() {
        let mut store = KvStore::new(8, nonce());
        let parent = store.create_working_set();
        prefill(&mut store, parent, &(1..=8).collect::<Vec<_>>(), &[8], 4);
        let parent_tail = store.lookup(parent, 1).unwrap();

        let child = store.fork(parent, Default::default()).unwrap();
        let ((copy_src, copy_dst), txn) = realize_declaration(&mut store, child, 1..2).unwrap();
        assert_eq!(copy_src, vec![parent_tail.0]);
        assert_eq!(copy_dst.len(), 1);
        assert_ne!(store.lookup(child, 1).unwrap(), parent_tail);
        assert_eq!(store.lookup(parent, 1).unwrap(), parent_tail);
        assert!(store.page_token_hashes(child, 1).unwrap().is_empty());
        finalize(&mut store, txn.unwrap(), true).unwrap();
    }

    fn prefill(store: &mut KvStore, ws: WorkingSetId, tokens: &[u32], fires: &[usize], page: u32) {
        let mut done = 0usize;
        for &n in fires {
            let chunk = &tokens[done..done + n];
            let (_, _, _, txn) = prepare(store, ws, done as u32, chunk, page, Some(chunk)).unwrap();
            finalize(store, txn, true).unwrap();
            done += n;
        }
    }

    fn translation_overlays_prepared_targets_on_the_committed_mapping() {
        let mut store = KvStore::new(16, nonce());
        let ws = store.create_working_set();
        let page = 4u32;

        let (proj, _, tr, txn) =
            prepare(&mut store, ws, 0, &[1, 2, 3, 4, 5, 6], page, None).unwrap();
        assert_eq!(tr, proj.physical_page_ids);
        finalize(&mut store, txn, true).unwrap();

        let forked = store.fork(ws, Default::default()).unwrap();
        let shared_head = store.lookup(forked, 0).unwrap().0;
        let shared_tail = store.lookup(forked, 1).unwrap().0;
        let (_, _, tr, txn) = prepare(&mut store, forked, 6, &[7], page, None).unwrap();
        assert_eq!(tr[0], shared_head);
        assert_ne!(tr[1], shared_tail);
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.lookup(forked, 1).unwrap().0, tr[1]);
    }
}
