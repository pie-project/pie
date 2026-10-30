use crate::store::kv::page_table::{PhysicalKvPageId, WorkingSetId};
use crate::store::kv::write::PageCommit;
use crate::store::kv::{KvStore, KvStoreError};

pub struct KvTxn {
    seq: u64,
}

pub type PageCopies = (Vec<u32>, Vec<u32>);

pub type CopiedBeforeWrite = (PageCopies, Option<KvTxn>);

/// A fire never writes a sealed page: sealed pages are frozen.
fn copied_before_write(
    store: &mut KvStore,
    ws: WorkingSetId,
    written: std::ops::Range<u64>,
) -> Result<Vec<u64>, KvStoreError> {
    let mapped = store.mapped_len(ws)?;
    let mut pages = Vec::new();
    for index in written.start..written.end.min(mapped) {
        if !store.privately_writable(ws, index)? {
            pages.push(index);
        }
    }
    Ok(pages)
}

pub fn written_demand(
    store: &mut KvStore,
    ws: WorkingSetId,
    written: std::ops::Range<u64>,
) -> Result<usize, KvStoreError> {
    let pages = copied_before_write(store, ws, written)?;
    if pages.is_empty() {
        return Ok(0);
    }
    store.write_demand(ws, &pages)
}

/// A copy keeps its hashes: its frozen slots hold what they held.
pub fn realize_written_reserved(
    store: &mut KvStore,
    ws: WorkingSetId,
    written: std::ops::Range<u64>,
    granted: &mut Vec<PhysicalKvPageId>,
) -> Result<CopiedBeforeWrite, KvStoreError> {
    let pages = copied_before_write(store, ws, written)?;
    if pages.is_empty() {
        return Ok(((Vec::new(), Vec::new()), None));
    }
    let prepared = store.prepare_write_reserved(ws, &pages, granted)?;
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
    let seq = store.publish_prepared(prepared, &commits)?;
    Ok((copies, Some(KvTxn { seq })))
}

pub fn prepare_explicit_reserved(
    store: &mut KvStore,
    ws: WorkingSetId,
    write_indexes: &[u64],
    granted: &mut Vec<PhysicalKvPageId>,
) -> Result<(PageCopies, KvTxn), KvStoreError> {
    let prepared = store.prepare_write_reserved(ws, write_indexes, granted)?;
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
    let seq = store.publish_prepared(prepared, &commits)?;
    Ok((copies, KvTxn { seq }))
}

pub fn settle(store: &mut KvStore, txn: KvTxn) {
    store.settle(txn.seq);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonce() -> [u8; 32] {
        [7u8; 32]
    }

    /// A fire copies the shared page of a fork's buffer it writes first,
    /// and the frozen page they share stays shared.
    #[test]
    fn a_fire_copies_the_shared_page_it_writes() {
        let mut store = KvStore::new(8, nonce());
        store.set_page_size(4);
        let parent = store.create_working_set();
        // Slots 0..4 frozen and sealed, 4..8 the buffer.
        prefill(&mut store, parent, &(1..=8).collect::<Vec<_>>(), 4);
        let parent_head = store.lookup(parent, 0).unwrap();
        let parent_tail = store.lookup(parent, 1).unwrap();

        let child = store.fork(parent).unwrap();
        assert_eq!(
            store.folded(child).unwrap(),
            4,
            "a fork keeps the frozen prefix"
        );
        assert_eq!(written_demand(&mut store, child, 1..2).unwrap(), 1);
        let mut granted = store.reserve_device_pages(1).unwrap();
        let ((copy_src, copy_dst), txn) =
            realize_written_reserved(&mut store, child, 1..2, &mut granted).unwrap();
        assert_eq!(copy_src, vec![parent_tail.0]);
        assert_eq!(copy_dst.len(), 1);
        assert_ne!(store.lookup(child, 1).unwrap(), parent_tail);
        assert_eq!(store.lookup(parent, 1).unwrap(), parent_tail);
        assert_eq!(store.lookup(child, 0).unwrap(), parent_head);
        settle(&mut store, txn.unwrap());
        assert_eq!(written_demand(&mut store, child, 1..2).unwrap(), 0);
    }

    fn prefill(store: &mut KvStore, ws: WorkingSetId, tokens: &[u32], page: u32) {
        let pages = (tokens.len() as u64).div_ceil(u64::from(page));
        store.reserve(ws, pages).unwrap();
        let mut granted = store.reserve_device_pages(pages as usize).unwrap();
        store
            .ensure_backed_reserved(ws, pages, &mut granted)
            .unwrap();
        store.log_rows(ws, 0, tokens).unwrap();
        assert_eq!(store.fold(ws, u64::from(page), false, None).unwrap(), 1);
    }
}
