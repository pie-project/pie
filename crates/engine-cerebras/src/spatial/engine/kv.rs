//! The KV cache as the tokens that made it, and the host tier's pages.

use engine::error::{Error, Result as EngineResult};

use super::{Spatial, launch};
use crate::spatial::wafer;

/// A suspended page on the host: the device page's words and its cells'
/// tokens.
type HostPage = (Vec<u32>, Option<Vec<Option<u32>>>);

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
    /// On a paged model: page cells record what the model's pages hold, so
    /// only paged fires (and copies) write them; slots whose pages the
    /// model never got (a fire that named another lane's pages) replay
    /// from their own tokens until they start over.
    pub(super) paged: bool,
    pub(super) tainted: std::collections::HashSet<u32>,
    /// The model's own cache: the tokens it holds, position by position,
    /// and the argmax after each where a token loop told it.
    pub(super) cached: Cached,
    /// The host tier: suspended pages by host page id, the device page's
    /// words and its cells' tokens; at most `host_pages` of them.
    pub(super) host: std::collections::HashMap<u32, HostPage>,
    pub(super) host_pages: u32,
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
                             engine wrote or copied: the spatial engine keeps a cache as the \
                             tokens that made it"
                        ))
                    })
            })
            .collect()
    }

    /// Whether the cells of `pages` below slot `upto` all hold a token, or
    /// are in `planned` (written earlier in the same fire).
    pub(super) fn known(
        &self,
        pages: &[u32],
        upto: u32,
        planned: &std::collections::HashSet<(u32, u32)>,
    ) -> bool {
        let size = self.page_size.max(1);
        (0..upto).all(|q| {
            let Some(&page) = pages.get((q / size) as usize) else {
                return false;
            };
            planned.contains(&(page, q % size))
                || self
                    .pages
                    .get(&page)
                    .and_then(|cells| cells.get((q % size) as usize).copied().flatten())
                    .is_some()
        })
    }

    /// A paged fire's rows: `tokens` at KV `slots` of `pages`, and the
    /// slot's own tokens where the rows continue them.
    pub(super) fn write_paged(&mut self, slot: u32, pages: &[u32], slots: &[u32], tokens: &[u32]) {
        let size = self.page_size.max(1);
        for (&q, &tok) in slots.iter().zip(tokens) {
            if let Some(&page) = pages.get((q / size) as usize) {
                let cells = self
                    .pages
                    .entry(page)
                    .or_insert_with(|| vec![None; size as usize]);
                cells[(q % size) as usize] = Some(tok);
            }
        }
        for page in pages {
            self.copied.remove(page);
        }
        let first = slots.first().copied().unwrap_or(0);
        let contiguous = slots
            .iter()
            .enumerate()
            .all(|(i, &q)| q == first + i as u32);
        match self.slots.get_mut(&slot) {
            Some(have) if contiguous && have.len() >= first as usize => {
                have.truncate(first as usize);
                have.extend_from_slice(tokens);
            }
            _ if contiguous && first == 0 => {
                self.slots.insert(slot, tokens.to_vec());
            }
            _ => {
                self.slots.remove(&slot);
            }
        }
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
        if pages.is_empty() || self.paged {
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

impl Spatial {
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

    /// A suspension's pages to the host tier: whole pages only (the host
    /// never attends a page; a lane parks and resumes whole).
    pub(super) fn page_out(&mut self, copy: &engine::transfer::KvCopy) -> EngineResult<()> {
        if !copy.moves.is_empty() || !copy.windowed.is_empty() {
            return Err(Error::Unsupported {
                verb: "`copy_kv` of part of a page to the host",
                engine: "cerebras-spatial",
            });
        }
        if let Some(&page) = copy.dst_page_ids.iter().find(|&&p| p >= self.kv.host_pages) {
            return Err(Error::Invalid(format!(
                "host page {page} of a host tier of {}",
                self.kv.host_pages
            )));
        }
        let Some(w) = self.model.as_mut() else {
            return Err(Error::Device("the host tier without a wafer model".into()));
        };
        let t0 = std::time::Instant::now();
        let words = w
            .page_out(copy.src_page_ids.clone())
            .map_err(|e| Error::Device(format!("pages to the host: {e}")))?;
        for ((&src, &dst), words) in copy.src_page_ids.iter().zip(&copy.dst_page_ids).zip(words) {
            let cells = self.kv.pages.get(&src).cloned();
            self.kv.host.insert(dst, (words, cells));
        }
        tracing::debug!(
            pages = copy.src_page_ids.len(),
            ms = t0.elapsed().as_secs_f64() * 1e3,
            "spatial: kv pages to the host"
        );
        Ok(())
    }

    /// A resumption's pages back from the host tier.
    pub(super) fn page_in(&mut self, copy: &engine::transfer::KvCopy) -> EngineResult<()> {
        if !copy.moves.is_empty() || !copy.windowed.is_empty() {
            return Err(Error::Unsupported {
                verb: "`copy_kv` of part of a page from the host",
                engine: "cerebras-spatial",
            });
        }
        let mut pages = Vec::with_capacity(copy.src_page_ids.len());
        for (&src, &dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            let (words, _) = self
                .kv
                .host
                .get(&src)
                .ok_or_else(|| Error::Invalid(format!("host page {src} holds nothing")))?;
            pages.push((dst, words.clone()));
        }
        let Some(w) = self.model.as_mut() else {
            return Err(Error::Device("the host tier without a wafer model".into()));
        };
        let t0 = std::time::Instant::now();
        w.page_in(pages)
            .map_err(|e| Error::Device(format!("pages from the host: {e}")))?;
        let size = self.kv.page_size.max(1) as usize;
        for (&src, &dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            let cells = self
                .kv
                .host
                .get(&src)
                .and_then(|(_, c)| c.clone())
                .unwrap_or_else(|| vec![None; size]);
            self.kv.pages.insert(dst, cells);
            self.kv.copied.insert(dst);
        }
        tracing::debug!(
            pages = copy.src_page_ids.len(),
            ms = t0.elapsed().as_secs_f64() * 1e3,
            "spatial: kv pages from the host"
        );
        Ok(())
    }
}
