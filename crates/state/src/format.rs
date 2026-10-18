//! How a class lays out its state, so a transfer between two layouts can be
//! planned and checked.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Planes {
    /// Keys and values, separately.
    KeyValue,
    /// One latent plane (MLA).
    Latent,
}

/// One engine's layout of a page of KV. A transfer is planned from one
/// format to another; two formats that agree on every field need no
/// conversion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateFormat {
    pub layers: u32,
    pub planes: Planes,
    /// KV heads in this shard.
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Tokens per page: a multiple of the cluster block.
    pub page_tokens: u32,
    /// The element type, such as `bf16`.
    pub dtype: String,
    /// Bits per element.
    pub dtype_bits: u32,
    /// Heads before tokens within a page (HND) rather than after (NHD).
    pub head_major: bool,
    /// This shard's rank and the group's size: heads are split evenly.
    pub rank: u32,
    pub ranks: u32,
}

impl StateFormat {
    /// Bytes one page holds across every layer and plane.
    pub fn page_bytes(&self) -> u64 {
        let planes = match self.planes {
            Planes::KeyValue => 2,
            Planes::Latent => 1,
        };
        let elements = u64::from(self.layers)
            * planes
            * u64::from(self.kv_heads)
            * u64::from(self.head_dim)
            * u64::from(self.page_tokens);
        (elements * u64::from(self.dtype_bits)).div_ceil(8)
    }
}

/// One KV plane of a page, as a host slot holds it: `[token][head][dim]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Plane {
    /// Every head the model has in this plane; a rank holds an even share.
    pub heads: u32,
    pub head_dim: u32,
    pub elem_bytes: u32,
}

impl Plane {
    fn heads_per_rank(&self, ranks: u32) -> u32 {
        (self.heads / ranks.max(1)).max(1)
    }

    /// The model head a rank's `local` head is, where heads split evenly
    /// across ranks (and a rank of a group wider than the heads holds a copy
    /// of one).
    fn head_of(&self, ranks: u32, rank: u32, local: u32) -> u32 {
        if self.heads >= ranks {
            rank * self.heads_per_rank(ranks) + local
        } else {
            rank * self.heads / ranks
        }
    }

    fn bytes(&self, ranks: u32, page_tokens: u32) -> usize {
        page_tokens as usize
            * self.heads_per_rank(ranks) as usize
            * self.head_dim as usize
            * self.elem_bytes as usize
    }
}

/// How an engine's host-tier read lays out pages: rank after rank, and within
/// a rank page after page, each page its planes in order. Two layouts of one
/// model that differ only in `ranks` hold the same state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageLayout {
    pub ranks: u32,
    pub page_tokens: u32,
    pub planes: Vec<Plane>,
}

impl PageLayout {
    fn rank_page_bytes(&self) -> usize {
        self.planes
            .iter()
            .map(|plane| plane.bytes(self.ranks, self.page_tokens))
            .sum()
    }

    /// Bytes `pages` pages take, across every rank.
    pub fn bytes(&self, pages: usize) -> usize {
        self.rank_page_bytes() * self.ranks as usize * pages
    }

    /// Whether `other` holds the same state another way: the same planes and
    /// page size, split across another number of ranks.
    pub fn convertible_to(&self, other: &PageLayout) -> bool {
        self.page_tokens == other.page_tokens && self.planes == other.planes
    }
}

/// `pages` pages laid out as `from`, laid out again as `to`: each head of
/// each token moves to the rank `to` gives it.
pub fn reshard(
    bytes: &[u8],
    pages: usize,
    from: &PageLayout,
    to: &PageLayout,
) -> Result<Vec<u8>, String> {
    if from == to {
        return Ok(bytes.to_vec());
    }
    if !from.convertible_to(to) {
        return Err("the layouts hold different state".into());
    }
    if bytes.len() != from.bytes(pages) {
        return Err(format!(
            "{} bytes are not {pages} pages of {}",
            bytes.len(),
            from.bytes(pages)
        ));
    }
    let tokens = from.page_tokens as usize;
    let (src_page, dst_page) = (from.rank_page_bytes(), to.rank_page_bytes());
    let src_rank = src_page * pages;
    let mut out = vec![0u8; to.bytes(pages)];
    for rank in 0..to.ranks {
        let mut plane_at_dst = 0usize;
        let mut plane_at_src = 0usize;
        for plane in &to.planes {
            let head = plane.head_dim as usize * plane.elem_bytes as usize;
            let (src_heads, dst_heads) = (
                plane.heads_per_rank(from.ranks) as usize,
                plane.heads_per_rank(to.ranks) as usize,
            );
            // This rank's heads as runs that sit side by side in one source
            // rank: `(first local here, source rank, first local there, heads)`.
            let mut runs: Vec<(usize, usize, usize, usize)> = Vec::new();
            for local in 0..dst_heads as u32 {
                let model_head = plane.head_of(to.ranks, rank, local);
                let (src, src_local) = if plane.heads >= from.ranks {
                    let per = plane.heads_per_rank(from.ranks);
                    (model_head / per, model_head % per)
                } else {
                    ((model_head * from.ranks).div_ceil(plane.heads), 0)
                };
                let (src, src_local) = (src as usize, src_local as usize);
                match runs.last_mut() {
                    Some((_, s, l, n)) if *s == src && *l + *n == src_local => *n += 1,
                    _ => runs.push((local as usize, src, src_local, 1)),
                }
            }
            for page in 0..pages {
                let dst_base = rank as usize * dst_page * pages + page * dst_page + plane_at_dst;
                for &(local, src, src_local, n) in &runs {
                    let src_base = src * src_rank + page * src_page + plane_at_src;
                    let len = n * head;
                    for token in 0..tokens {
                        let s = src_base + (token * src_heads + src_local) * head;
                        let d = dst_base + (token * dst_heads + local) * head;
                        out[d..d + len].copy_from_slice(&bytes[s..s + len]);
                    }
                }
            }
            plane_at_src += plane.bytes(from.ranks, tokens as u32);
            plane_at_dst += plane.bytes(to.ranks, tokens as u32);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(ranks: u32) -> PageLayout {
        PageLayout {
            ranks,
            page_tokens: 2,
            planes: vec![
                Plane {
                    heads: 4,
                    head_dim: 2,
                    elem_bytes: 1,
                },
                Plane {
                    heads: 4,
                    head_dim: 2,
                    elem_bytes: 1,
                },
            ],
        }
    }

    /// Each element names its page, plane, token, model head and lane, so a
    /// misplaced byte shows where it came from.
    fn fill(layout: &PageLayout, pages: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for rank in 0..layout.ranks {
            for page in 0..pages {
                for (p, plane) in layout.planes.iter().enumerate() {
                    for token in 0..layout.page_tokens {
                        for local in 0..plane.heads_per_rank(layout.ranks) {
                            let head = plane.head_of(layout.ranks, rank, local);
                            for lane in 0..plane.head_dim {
                                out.push(
                                    (page * 64
                                        + p * 32
                                        + token as usize * 8
                                        + head as usize * 2
                                        + lane as usize) as u8,
                                );
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn heads_regroup_across_ranks_and_come_back() {
        let (one, two, four) = (layout(1), layout(2), layout(4));
        let whole = fill(&one, 3);
        assert_eq!(whole.len(), one.bytes(3));
        let split = reshard(&whole, 3, &one, &two).unwrap();
        assert_eq!(split, fill(&two, 3));
        assert_eq!(reshard(&split, 3, &two, &four).unwrap(), fill(&four, 3));
        assert_eq!(reshard(&split, 3, &two, &one).unwrap(), whole);
    }

    #[test]
    fn other_state_is_refused() {
        let mut other = layout(1);
        other.planes[0].head_dim = 4;
        assert!(reshard(&fill(&layout(1), 1), 1, &layout(1), &other).is_err());
    }
}
