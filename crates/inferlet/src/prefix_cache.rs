//! A prefix cache an inferlet keeps over the working-set index: page `k` of
//! a prompt is published under `namespace || h_k`, where `h_k` chains the
//! hashes of the prompt's first `k` pages, so a later prompt with the same
//! leading pages maps them in instead of prefilling them.

use crate::eta::{Pipeline, WorkingSet, kv_page_size};
use crate::model::{self, ForwardKind};

pub struct PrefixCache {
    namespace: Vec<u8>,
    page_size: u32,
    enabled: bool,
}

impl PrefixCache {
    /// `namespace` separates programs: keys are global to the model's store,
    /// so it must name everything the published KV depends on besides tokens.
    pub fn new(namespace: &[u8]) -> PrefixCache {
        // Only models whose context lives in full KV pages: recurrent state
        // has no index, and gemma4's windowed rows are claimed per tail, so a
        // page boundary behind the tail would come back without them.
        let enabled = model::pass_kind() == ForwardKind::Attention
            && model::rs_state_size() == 0
            && model::architecture() != "gemma4";
        PrefixCache {
            namespace: namespace.to_vec(),
            page_size: kv_page_size(),
            enabled,
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

    /// The working set holding the longest published page-aligned prefix of
    /// `tokens` (always leaving the last token to compute) and the tokens it
    /// covers; an empty set and 0 when nothing matches.
    pub fn adopt(&self, tokens: &[u32]) -> Result<(WorkingSet, u32), String> {
        let max = if self.enabled {
            tokens.len().saturating_sub(1) as u32 / self.page_size
        } else {
            0
        };
        let keys = self.keys(tokens, max);
        // Publishers index every page up to their prompt's end, so the chain
        // is prefix-closed and a binary search finds the longest hit; an
        // evicted boundary only makes it settle on a shorter one.
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
        Ok(match best {
            Some(ws) => (ws, lo * self.page_size),
            None => (WorkingSet::new(), 0),
        })
    }

    /// Publishes every full page of `tokens` past the first `from` tokens.
    /// `ws` must hold exactly the model's KV of `tokens` from a settled,
    /// unmasked prefill.
    pub fn publish(
        &self,
        ws: &WorkingSet,
        on: &Pipeline,
        tokens: &[u32],
        from: u32,
    ) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        let pages = tokens.len() as u32 / self.page_size;
        let keys = self.keys(tokens, pages);
        for k in from / self.page_size + 1..=pages {
            ws.slice(on, 0, k)?.update_index(&keys[k as usize - 1])?;
        }
        Ok(())
    }
}
