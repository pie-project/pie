pub type BlockId = u32;

pub type PhysicalPageId = BlockId;

#[derive(Debug, PartialEq, Eq)]
pub enum PrepareError {
    StaleGeneration {
        captured: u32,
        current: u32,
    },
    InvalidValidLen {
        index: u32,
        valid_len: u32,
        page_size: u32,
    },
    DuplicateOutputIndex(u32),
    NonContiguousActiveRun {
        gap_at: u32,
    },
    EmptyForward,
    NoInputTokens,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWrite {
    pub slot_index: u32,
    pub page: PhysicalPageId,
    pub valid_len: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub struct KvProjection {
    pub physical_page_ids: Vec<PhysicalPageId>,
    pub last_page_len: u32,
    pub active_page_idx: Option<usize>,
    pub full_page_writes: Vec<u32>,
}

pub fn project_kv(
    context_pages: &[PhysicalPageId],
    context_valid_tokens: u32,
    writes: &[KvWrite],
    page_size: u32,
) -> Result<KvProjection, PrepareError> {
    if context_pages.is_empty() && writes.is_empty() {
        return Err(PrepareError::EmptyForward);
    }

    let context_len = context_pages.len() as u32;
    let mut max_write_slot: Option<u32> = None;
    let mut seen: Vec<u32> = Vec::with_capacity(writes.len());
    for wr in writes {
        if wr.valid_len == 0 || wr.valid_len > page_size {
            return Err(PrepareError::InvalidValidLen {
                index: wr.slot_index,
                valid_len: wr.valid_len,
                page_size,
            });
        }
        if seen.contains(&wr.slot_index) {
            return Err(PrepareError::DuplicateOutputIndex(wr.slot_index));
        }
        seen.push(wr.slot_index);
        max_write_slot = Some(max_write_slot.map_or(wr.slot_index, |m| m.max(wr.slot_index)));
    }

    let active_len = context_len.max(max_write_slot.map_or(0, |m| m + 1));

    let mut physical_page_ids = Vec::with_capacity(active_len as usize);
    for slot in 0..active_len {
        if let Some(wr) = writes.iter().find(|w| w.slot_index == slot) {
            physical_page_ids.push(wr.page);
        } else if slot < context_len {
            physical_page_ids.push(context_pages[slot as usize]);
        } else {
            return Err(PrepareError::NonContiguousActiveRun { gap_at: slot });
        }
    }

    let last_slot = active_len - 1;
    let last_page_len = if let Some(wr) = writes.iter().find(|w| w.slot_index == last_slot) {
        wr.valid_len
    } else {
        let consumed = last_slot * page_size;
        let rem = context_valid_tokens.saturating_sub(consumed);
        if rem == 0 || rem > page_size {
            page_size
        } else {
            rem
        }
    };

    let active_page_idx = max_write_slot.map(|m| m as usize);

    let full_page_writes = writes
        .iter()
        .filter(|w| w.valid_len == page_size)
        .map(|w| w.slot_index)
        .collect();

    Ok(KvProjection {
        physical_page_ids,
        last_page_len,
        active_page_idx,
        full_page_writes,
    })
}
