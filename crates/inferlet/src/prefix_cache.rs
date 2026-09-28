//! A prefix cache an inferlet keeps over the working-set indexes: a prompt
//! prefix of `k` pages is published under `namespace || h_k`, where `h_k`
//! chains the hashes of the prompt's first `k` pages, so a later prompt with
//! the same leading pages maps them in instead of prefilling them.
//!
//! KV pages can be sliced out of a finished prompt, so an attention model
//! publishes every page. Recurrent state exists only where a prefill chunk
//! ended and a snapshot is a whole state slot, so a hybrid model publishes
//! only at boundaries its prefill ends chunks at, each under both indexes.

use crate::eta::{Pipeline, RsWorkingSet, WorkingSet, kv_page_size};
use crate::model::{self, ForwardKind};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Pages,
    Boundaries,
}

pub struct PrefixCache {
    namespace: Vec<u8>,
    page_size: u32,
    mode: Mode,
}

impl PrefixCache {
    /// `namespace` separates programs: keys are global to the model's store,
    /// so it must name everything the published state depends on besides tokens.
    pub fn new(namespace: &[u8]) -> PrefixCache {
        // gemma4's windowed rows are claimed per tail, so a page boundary
        // behind the tail would come back without them.
        let mode = match model::pass_kind() {
            _ if model::architecture() == "gemma4" => Mode::Off,
            ForwardKind::Attention => Mode::Pages,
            ForwardKind::Hybrid => Mode::Boundaries,
            ForwardKind::Recurrent | ForwardKind::Diffusion => Mode::Off,
        };
        PrefixCache {
            namespace: namespace.to_vec(),
            page_size: kv_page_size(),
            mode,
        }
    }

    fn keys(&self, tokens: &[u32], pages: u32) -> Vec<Vec<u8>> {
        let mut h = [0u8; 32];
        tokens
            .chunks_exact(self.page_size as usize)
            .take(pages as usize)
            .map(|page| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(&h);
                for t in page {
                    hasher.update(&t.to_le_bytes());
                }
                h = *hasher.finalize().as_bytes();
                [self.namespace.as_slice(), &h].concat()
            })
            .collect()
    }

    /// The positions past `from` a hybrid prefill must end a chunk at and
    /// hand `publish_state` the state of: `structure` (e.g. message ends) floored to
    /// pages, else pages 1, 2, 4, 8, ..., so O(log n) snapshots still reuse
    /// at least half of any shared prefix. Empty for an attention model.
    pub fn boundaries(&self, tokens: &[u32], from: u32, structure: Option<&[u32]>) -> Vec<u32> {
        if self.mode != Mode::Boundaries {
            return Vec::new();
        }
        // Always leave the last token to compute.
        let limit = tokens.len().saturating_sub(1) as u32;
        let mut cuts: Vec<u32> = match structure {
            Some(ends) => ends
                .iter()
                .map(|&end| end.min(limit) / self.page_size * self.page_size)
                .collect(),
            None => std::iter::successors(Some(self.page_size), |&b| b.checked_mul(2))
                .take_while(|&b| b <= limit)
                .collect(),
        };
        cuts.retain(|&b| b > from && b > 0);
        cuts.sort_unstable();
        cuts.dedup();
        cuts
    }

    /// The longest published prefix of `tokens`: its KV working set, its
    /// recurrent state on a hybrid model, and the tokens it covers; an empty
    /// set, none and 0 when nothing matches. `structure` must be what the
    /// publishers passed to `boundaries`.
    pub fn adopt(
        &self,
        tokens: &[u32],
        structure: Option<&[u32]>,
    ) -> Result<(WorkingSet, Option<RsWorkingSet>, u32), String> {
        let miss = || (WorkingSet::new(), None, 0);
        match self.mode {
            Mode::Off => Ok(miss()),
            Mode::Pages => {
                let max = tokens.len().saturating_sub(1) as u32 / self.page_size;
                let keys = self.keys(tokens, max);
                // Publishers index every page up to their prompt's end, so the
                // chain is prefix-closed and a binary search finds the longest
                // hit; an evicted boundary only makes it settle on a shorter one.
                let (mut lo, mut hi) = (0, max);
                let mut best = None;
                while lo < hi {
                    let mid = (lo + hi).div_ceil(2);
                    match WorkingSet::from_index(&keys[mid as usize - 1])? {
                        Some(ws) => {
                            best = Some(ws);
                            lo = mid;
                        }
                        None => hi = mid - 1,
                    }
                }
                Ok(best.map_or_else(miss, |ws| (ws, None, lo * self.page_size)))
            }
            Mode::Boundaries => {
                let cuts = self.boundaries(tokens, 0, structure);
                let keys = self.keys(tokens, cuts.last().map_or(0, |b| b / self.page_size));
                // Either half can be reclaimed alone; a boundary needs both.
                for &b in cuts.iter().rev() {
                    let key = &keys[(b / self.page_size) as usize - 1];
                    if let Some(ws) = WorkingSet::from_index(key)?
                        && let Some(rs) = RsWorkingSet::from_index(key)?
                    {
                        return Ok((ws, Some(rs), b));
                    }
                }
                Ok(miss())
            }
        }
    }

    /// Publishes every full page of `tokens` past the first `from` on an
    /// attention model. `ws` must hold exactly the model's KV of `tokens`
    /// from a settled, unmasked prefill.
    pub fn publish(
        &self,
        ws: &WorkingSet,
        on: &Pipeline,
        tokens: &[u32],
        from: u32,
    ) -> Result<(), String> {
        if self.mode != Mode::Pages {
            return Ok(());
        }
        let pages = tokens.len() as u32 / self.page_size;
        let keys = self.keys(tokens, pages);
        for k in from / self.page_size + 1..=pages {
            ws.slice(on, 0, k)?.update_index(&keys[k as usize - 1])?;
        }
        Ok(())
    }

    /// Publishes `tokens`, which end at one of `boundaries`, on a hybrid
    /// model, as soon as the chunk ending there has landed: `ws` and `rs`
    /// hold exactly their state. The snapshot belongs to the index, not to
    /// this process, so idle reclaim can take it back when slots run short.
    pub fn publish_state(
        &self,
        ws: &WorkingSet,
        rs: &RsWorkingSet,
        on: &Pipeline,
        tokens: &[u32],
    ) -> Result<(), String> {
        if self.mode != Mode::Boundaries {
            return Ok(());
        }
        let k = tokens.len() as u32 / self.page_size;
        let key = &self.keys(tokens, k)[k as usize - 1];
        rs.update_index(key)?;
        ws.slice(on, 0, k)?.update_index(key)
    }
}
