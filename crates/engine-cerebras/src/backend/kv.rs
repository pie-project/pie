//! The KV cache as the tokens that made it.

use engine::error::{Error, Result as EngineResult};

use super::{CerebrasEngine, launch};
use crate::wafer;

/// What the KV cache holds, as tokens: a one-sequence wafer program keeps
/// one sequence's cache, so every lane's cache is kept here as the tokens
/// that made it — by slot, and by pool page and offset for what `copy_kv`
/// moves between pages — and the model's cache is brought to a lane's
/// history before its rows run: kept as far as it agrees, cut there
/// (`truncate`), and the rest replayed.
///
/// A lane's history is its slot's, unless the slot holds too little or a
/// copy landed in the lane's pages since (a fork, a shared prefix): then
/// its pages'. (Two lanes of one fire may name the same pool pages, so a
/// page's cells alone do not say whose a cache is.)

#[derive(Default)]
pub(super) struct Kv {
    pub(super) page_size: u32,
    /// Pool page -> the token at each offset (`None`: never written).
    pub(super) pages: std::collections::HashMap<u32, Vec<Option<u32>>>,
    /// The slot's tokens.
    pub(super) slots: std::collections::HashMap<u32, Vec<u32>>,
    /// Pages a copy wrote since a lane last fired on them.
    pub(super) copied: std::collections::HashSet<u32>,
    /// The model's own cache: the tokens it holds, position by position,
    /// and the argmax after each where a token loop told it.
    pub(super) cached: Cached,
}

impl Kv {
    /// The lane's first `n` positions' tokens.
    pub(super) fn history(&self, slot: u32, pages: &[u32], n: u32) -> EngineResult<Vec<u32>> {
        let size = self.page_size.max(1);
        let copied_in = pages
            .iter()
            .take(n.div_ceil(size) as usize)
            .any(|page| self.copied.contains(page));
        let slot_holds = self.slots.get(&slot).is_some_and(|h| h.len() >= n as usize);
        if pages.is_empty() || (slot_holds && !copied_in) {
            let have = self.slots.get(&slot).map_or(&[][..], Vec::as_slice);
            return have.get(..n as usize).map(<[u32]>::to_vec).ok_or_else(|| {
                Error::Invalid(format!(
                    "slot {slot} fires at position {n} and the spatial engine holds {} of its \
                     tokens",
                    have.len()
                ))
            });
        }
        (0..n)
            .map(|p| {
                let page = pages.get((p / size) as usize).copied();
                page.and_then(|page| self.pages.get(&page))
                    .and_then(|cells| cells.get((p % size) as usize).copied().flatten())
                    .ok_or_else(|| {
                        Error::Invalid(format!(
                            "position {p} of slot {slot} (page {page:?}) holds no token this \
                             engine wrote or copied: the cerebras engine keeps a cache as the \
                             tokens that made it"
                        ))
                    })
            })
            .collect()
    }

    /// `tokens` written at positions `start..` of the lane.
    pub(super) fn write(
        &mut self,
        slot: u32,
        pages: &[u32],
        history: Vec<u32>,
        tokens: &[u32],
    ) -> EngineResult<()> {
        let start = history.len() as u32;
        let mut have = history;
        have.extend_from_slice(tokens);
        self.slots.insert(slot, have);
        for page in pages {
            self.copied.remove(page);
        }
        if pages.is_empty() {
            return Ok(());
        }
        let size = self.page_size.max(1);
        for (i, &tok) in tokens.iter().enumerate() {
            let p = start + i as u32;
            let page = *pages.get((p / size) as usize).ok_or_else(|| {
                Error::Invalid(format!(
                    "slot {slot} writes position {p} past its {} page(s)",
                    pages.len()
                ))
            })?;
            let cells = self
                .pages
                .entry(page)
                .or_insert_with(|| vec![None; size as usize]);
            cells[(p % size) as usize] = Some(tok);
        }
        Ok(())
    }
}

#[derive(Default, Clone)]
pub(super) struct Cached {
    pub(super) tokens: Vec<u32>,
    pub(super) argmax: Vec<Option<u32>>,
    /// Steps the next token loop on this cache runs past its row: it
    /// doubles while loops are used to their end and starts over at
    /// [`FIRST_LOOP`] otherwise, so a sequence that ends wastes at most
    /// about what it used.
    pub(super) next_loop: usize,
}

impl Cached {
    pub(super) fn truncate(&mut self, n: usize) {
        self.tokens.truncate(n);
        self.argmax.truncate(n);
    }

    pub(super) fn extend(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.argmax.resize(self.tokens.len(), None);
    }
}

impl Kv {
    /// How many of `history`'s first positions the model's cache agrees
    /// with.
    pub(super) fn agreement(&self, history: &[u32]) -> usize {
        self.cached
            .tokens
            .iter()
            .zip(history)
            .take_while(|(a, b)| a == b)
            .count()
    }
}

impl CerebrasEngine {
    /// Brings the model's cache, which agrees with `history` for its first
    /// `agree` positions, to `history`: the rest cut (another sequence, a
    /// rewind or rejected rows; a request at position n attends below it
    /// alone and writes over the rest) and `history`'s rest replayed.
    pub(super) fn align_cache(&mut self, agree: usize, history: &[u32]) -> EngineResult<()> {
        self.kv.cached.truncate(agree);
        if agree < history.len() {
            self.replayed += (history.len() - agree) as u64;
            tracing::info!(
                rows = history.len() - agree,
                total = self.replayed,
                "spatial: history replayed"
            );
            launch::forward(
                self.model()?,
                &history[agree..],
                agree as u32,
                wafer::Logits::None,
            )?;
            self.kv.cached.extend(&history[agree..]);
        }
        Ok(())
    }
}
