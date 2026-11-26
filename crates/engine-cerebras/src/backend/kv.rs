//! The KV cache as the tokens that made it.

use engine::error::{Error, Result as EngineResult};

use super::{CerebrasEngine, launch};
use crate::wafer;

/// What the KV cache holds, as tokens: every lane's cache is kept as the tokens that made it (by
/// slot, and by page for `copy_kv`), and the model's one-sequence cache is brought to a lane's
/// history before its rows run: kept as far as it agrees, the rest replayed. A lane's history is
/// its pages', its slot's where it has none.

#[derive(Default)]
pub(super) struct Kv {
    pub(super) page_size: u32,
    /// Kv list slot ref -> the token at each offset (`None`: never written).
    pub(super) pages: std::collections::HashMap<u32, Vec<Option<u32>>>,
    /// The slot's tokens.
    pub(super) slots: std::collections::HashMap<u32, Vec<u32>>,
    /// The model's own cache: its tokens by position, and the argmax after each where a token loop
    /// told it.
    pub(super) cached: Cached,
}

impl Kv {
    /// The kv list slot refs a lane's table names.
    pub(super) fn lane_pages(lane: &engine::fire::Lane) -> Vec<u32> {
        lane.tables
            .iter()
            .find(|(shape, _)| *shape == super::pool::KV)
            .map_or_else(Vec::new, |(_, refs)| refs.clone())
    }

    /// The lane's first `n` positions' tokens.
    pub(super) fn history(&self, slot: u32, pages: &[u32], n: u32) -> EngineResult<Vec<u32>> {
        let size = self.page_size.max(1);
        // The pages say what a lane's positions hold: a slot's next lane may start on a prefix
        // another wrote. A lane of no pages, or one whose pages were not written, is its slot's.
        let mut have: Vec<u32> = Vec::with_capacity(n as usize + 1);
        for page in pages.iter().take(n.div_ceil(size) as usize) {
            let want = (n as usize - have.len()).min(size as usize);
            let cells = self.pages.get(page).map_or(&[][..], Vec::as_slice);
            let written = cells.iter().take(want).map_while(|&t| t);
            let before = have.len();
            have.extend(written);
            if have.len() < before + want {
                break;
            }
        }
        if have.len() == n as usize {
            return Ok(have);
        }
        let at = have.len();
        let slot_holds = self.slots.get(&slot).map_or(&[][..], Vec::as_slice);
        slot_holds
            .get(..n as usize)
            .map(<[u32]>::to_vec)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "position {at} of slot {slot} (page {:?}; a fire at {n}) holds no token this \
                     engine wrote or copied: the cerebras engine keeps a cache as the tokens that \
                     made it",
                    pages.get(at / size as usize)
                ))
            })
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
    /// Steps the next token loop on this cache runs past its row: it doubles while loops are used
    /// to their end and starts over at [`FIRST_LOOP`] otherwise.
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
    /// How many of `history`'s first positions the model's cache agrees with.
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
    /// Brings the model's cache, which agrees with `history` for its first `agree` positions, to
    /// `history`: the rest cut and `history`'s rest replayed.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slots_next_lane_holds_the_prefix_its_pages_hold() {
        let mut kv = Kv {
            page_size: 2,
            ..Kv::default()
        };
        kv.write(0, &[7, 8], Vec::new(), &[1, 2, 3]).unwrap();
        kv.write(1, &[9, 5], Vec::new(), &[4, 5, 6, 7]).unwrap();
        // slot 1's next lane on slot 0's first page
        assert_eq!(kv.history(1, &[7, 6], 2).ok(), Some(vec![1, 2]));
        assert_eq!(kv.history(0, &[7, 8], 3).ok(), Some(vec![1, 2, 3]));
        // no pages: the slot's
        assert_eq!(kv.history(1, &[], 3).ok(), Some(vec![4, 5, 6]));
        assert!(kv.history(2, &[7, 6], 3).is_err());
    }
}
