//! Generalised tiktoken-style BPE merge.
//!
//! One merge algorithm, one data structure, no special-casing.
//!
//! The merge loop uses a **flat-array doubly-linked list** for O(1) merge
//! operations with O(1) predecessor access.  An adaptive dispatch chooses
//! between linear scan (short pieces) and BinaryHeap (long pieces).
//!
//! `BpeTable` uses *token-ID-pair* merge keys: merges are looked up as
//! `(left_id, right_id) → (rank, merged_id)`, avoiding variable-length
//! byte hashing entirely.  Symbol lookup (`bytes → TokenId`) is separate.

use std::collections::HashMap;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Token ID (the value returned to the caller).
pub type TokenId = u32;

/// Merge rank (lower = higher priority). Internal to the BPE algorithm.
type Rank = u32;

// ---------------------------------------------------------------------------
// BpeTable — the only data structure
// ---------------------------------------------------------------------------

pub struct BpeTable {
    /// Symbol lookup: bytes → token ID.
    token_to_id: FxHashMap<Vec<u8>, TokenId>,
    /// Merge lookup: (left_id, right_id) → (rank, merged_id).
    merges: FxHashMap<(TokenId, TokenId), (Rank, TokenId)>,
    /// Decode table: token_id → bytes (indexed by ID).
    id_to_bytes: Vec<Vec<u8>>,
    /// Pre-computed byte fallback: byte → token_id for `<0xNN>` tokens.
    byte_fallback_ids: [Option<TokenId>; 256],
}

impl BpeTable {
    /// Build from a rank→bytes map (tiktoken-native: rank == token ID).
    pub fn from_decoder_map(map: HashMap<TokenId, Vec<u8>>) -> Self {
        let max_id = map.keys().copied().max().unwrap_or(0) as usize;
        let mut id_to_bytes = vec![Vec::new(); max_id + 1];
        let mut token_to_id =
            FxHashMap::with_capacity_and_hasher(map.len(), Default::default());
        let mut merges = FxHashMap::default();

        // First pass: register all symbols.
        for (&id, bytes) in &map {
            id_to_bytes[id as usize] = bytes.clone();
            token_to_id.insert(bytes.clone(), id);
        }

        // Second pass: for every token of length ≥ 2, try all possible splits
        // to find which pair merges into it.  For tiktoken, rank == id.
        // The merge with the lowest resulting id wins.
        for (&id, bytes) in &map {
            if bytes.len() < 2 {
                continue;
            }
            // Try all split points.
            for split in 1..bytes.len() {
                let left = &bytes[..split];
                let right = &bytes[split..];
                if let (Some(&left_id), Some(&right_id)) =
                    (token_to_id.get(left), token_to_id.get(right))
                {
                    let rank = id; // tiktoken: rank == id
                    merges
                        .entry((left_id, right_id))
                        .and_modify(|e: &mut (Rank, TokenId)| {
                            if rank < e.0 {
                                *e = (rank, id);
                            }
                        })
                        .or_insert((rank, id));
                }
            }
        }

        BpeTable {
            token_to_id,
            merges,
            id_to_bytes,
            byte_fallback_ids: [None; 256],
        }
    }

    /// Build from HF `tokenizer.json` vocab + merges.
    ///
    /// Merge priority comes from position in the merges array.
    /// Single-character vocab entries are atoms (not merge results).
    ///
    /// If `raw_byte_keys` is true, GPT-2 unicode keys in `vocab` are
    /// converted to raw bytes via the inverse byte mapping.  This lets
    /// the encode path work directly on `&[u8]` without running
    /// `bytes_to_unicode()`.
    pub fn from_vocab_and_merges(
        vocab: &HashMap<String, u32>,
        merge_pairs: &[(String, String)],
        continuing_subword_prefix: &str,
        raw_byte_keys: bool,
    ) -> Self {
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_bytes = vec![Vec::new(); max_id + 1];
        let mut token_to_id =
            FxHashMap::with_capacity_and_hasher(vocab.len(), Default::default());
        let mut merges = FxHashMap::default();

        // First pass: register all symbols.
        for (token, &id) in vocab {
            let key = if raw_byte_keys {
                gpt2_unicode_to_raw_bytes(token)
            } else {
                token.as_bytes().to_vec()
            };
            id_to_bytes[id as usize] = key.clone();
            token_to_id.entry(key).or_insert(id);
        }

        // Second pass: build merge table from explicit merge pairs.
        for (idx, (a, b)) in merge_pairs.iter().enumerate() {
            let a_key = if raw_byte_keys {
                gpt2_unicode_to_raw_bytes(a)
            } else {
                a.as_bytes().to_vec()
            };

            // Look up b's token ID using the stripped key.
            // At runtime, lower-level merges produce the *stripped* token (e.g. "he"
            // not "▁he"), so the right_id must be the stripped token's ID.
            let b_stripped = if !continuing_subword_prefix.is_empty()
                && b != continuing_subword_prefix
            {
                match b.strip_prefix(continuing_subword_prefix) {
                    Some(stripped) => stripped,
                    None => b.as_str(),
                }
            } else {
                b.as_str()
            };
            let b_key = if raw_byte_keys {
                gpt2_unicode_to_raw_bytes(b_stripped)
            } else {
                b_stripped.as_bytes().to_vec()
            };

            // Merged bytes = a_key + b_key (prefix removed from b).
            let mut merged_key = a_key.clone();
            merged_key.extend_from_slice(&b_key);

            if let (Some(&left_id), Some(&right_id), Some(&merged_id)) = (
                token_to_id.get(&a_key),
                token_to_id.get(&b_key),
                token_to_id.get(&merged_key),
            ) {
                let rank = (idx + 1) as Rank;
                // First occurrence wins (lower rank = higher priority).
                merges.entry((left_id, right_id)).or_insert((rank, merged_id));
            }
        }

        // Pre-compute byte fallback table: byte → token_id of `<0xNN>`.
        let mut byte_fallback_ids = [None; 256];
        for byte in 0u16..=255 {
            let hex = format!("<0x{byte:02X}>");
            if let Some(&id) = vocab.get(&hex) {
                byte_fallback_ids[byte as usize] = Some(id);
            }
        }

        BpeTable {
            token_to_id,
            merges,
            id_to_bytes,
            byte_fallback_ids,
        }
    }

    /// Look up token ID for a byte sequence.
    #[inline]
    pub fn bytes_to_id(&self, bytes: &[u8]) -> Option<TokenId> {
        self.token_to_id.get(bytes).copied()
    }

    /// Look up bytes for a token ID.
    #[inline]
    pub fn id_to_bytes(&self, id: TokenId) -> Option<&[u8]> {
        self.id_to_bytes
            .get(id as usize)
            .filter(|v| !v.is_empty())
            .map(|v| v.as_slice())
    }

    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Insert a token (for added/special tokens).
    ///
    /// Does NOT add merge entries — added tokens are matched by AhoCorasick
    /// before BPE encoding, so they never participate in the merge algorithm.
    /// Skips `token_to_id` insertion if the ID already exists in the base vocab
    /// (some models list tokens in both `vocab` and `added_tokens`).
    pub fn insert(&mut self, bytes: Vec<u8>, id: TokenId) {
        if id as usize >= self.id_to_bytes.len() {
            self.id_to_bytes.resize(id as usize + 1, Vec::new());
        }
        // Only insert if this ID doesn't already have a mapping.
        // This prevents duplicate entries when added_tokens overlap the base
        // vocab (e.g. DeepSeek-V3.2 where GPT-2 byte remapping produces
        // different key bytes for the same token).
        if self.id_to_bytes[id as usize].is_empty() {
            self.id_to_bytes[id as usize] = bytes.clone();
            self.token_to_id.entry(bytes).or_insert(id);
        }
    }

    /// Pair-merge rank lookup: can (left, right) merge?
    #[inline]
    fn pair_rank(&self, left: TokenId, right: TokenId) -> Rank {
        self.merges
            .get(&(left, right))
            .map_or(Rank::MAX, |&(r, _)| r)
    }

    /// Pair-merge lookup: returns (rank, merged_id).
    #[inline]
    fn pair_merge(&self, left: TokenId, right: TokenId) -> Option<(Rank, TokenId)> {
        self.merges.get(&(left, right)).copied()
    }

    /// Byte-fallback lookup (internal).
    #[inline]
    fn byte_fallback(&self, byte: u8) -> Option<TokenId> {
        self.byte_fallback_ids[byte as usize]
    }
}
// ---------------------------------------------------------------------------
// GPT-2 byte ↔ unicode mapping (compile-time const tables)
// ---------------------------------------------------------------------------

/// Determine whether byte `b` maps directly (identity) in the GPT-2 scheme.
/// These are printable ASCII (0x21..=0x7E) plus Latin-1 Supplement (0xA1..=0xAC, 0xAE..=0xFF).
const fn is_direct_byte(b: u8) -> bool {
    matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)
}

/// Build the GPT-2 byte→unicode table at compile time.
///
/// Direct bytes map to their codepoint (identity).
/// Non-direct bytes (control chars, 0x7F, 0xAD, etc.) map to U+0100 + offset.
const fn build_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    let mut b = 0u16;
    while b < 256 {
        if is_direct_byte(b as u8) {
            // SAFETY: direct bytes are valid Unicode code points (0x21..0xFF)
            table[b as usize] = match char::from_u32(b as u32) {
                Some(c) => c,
                None => '\0', // unreachable for valid direct bytes
            };
        } else {
            table[b as usize] = match char::from_u32(256 + n) {
                Some(c) => c,
                None => '\0', // unreachable, 256+n always valid
            };
            n += 1;
        }
        b += 1;
    }
    table
}

/// Build the GPT-2 char→byte (inverse) table at compile time.
///
/// The max unicode code point used is 256 + 33 - 1 = 288 for non-direct bytes,
/// plus direct bytes up to 0xFF = 255.  Max index = 288, so we need size 324
/// to match the original table (with some headroom).
const fn build_char_to_byte() -> [Option<u8>; 324] {
    let b2u = build_byte_to_unicode();
    let mut table: [Option<u8>; 324] = [None; 324];
    let mut b = 0u16;
    while b < 256 {
        let idx = b2u[b as usize] as usize;
        if idx < 324 {
            table[idx] = Some(b as u8);
        }
        b += 1;
    }
    table
}

/// GPT-2 byte→unicode mapping table (256 entries, one char per byte).
///
/// Used by the ByteLevel pre-tokenizer to remap raw bytes into GPT-2
/// unicode chars before applying the regex, ensuring identical splits
/// as the HuggingFace `tokenizers` library.
static BYTE_TO_UNICODE: [char; 256] = build_byte_to_unicode();

/// GPT-2 char→byte (inverse) mapping table.
static CHAR_TO_BYTE: [Option<u8>; 324] = build_char_to_byte();

/// Get the GPT-2 byte→unicode mapping table.
pub fn bytes_to_unicode() -> &'static [char; 256] {
    &BYTE_TO_UNICODE
}

/// Convert a GPT-2 unicode token string to raw bytes.
///
/// Used at load time to re-key the vocabulary.  Each GPT-2 unicode char
/// maps 1:1 to a byte value.
fn gpt2_unicode_to_raw_bytes(token: &str) -> Vec<u8> {
    token
        .chars()
        .map(|c| {
            let idx = c as usize;
            if idx < 324 {
                CHAR_TO_BYTE[idx].unwrap_or(c as u8)
            } else {
                // Non-GPT2 char: keep raw encoding.  Should not happen
                // for well-formed GPT-2 vocabs.
                c as u8
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Core: doubly-linked-list BPE merge (token-ID pair keys)
// ---------------------------------------------------------------------------

/// A node in the flat-array doubly-linked list used for BPE merging.
///
/// Each node represents a token in the current merge state.
/// `rank` caches the merge rank of fusing this node with its successor.
struct Node {
    token_id: TokenId,
    rank: Rank,
    prev: u32, // NONE = no predecessor (head)
    next: u32, // NONE = no successor (sentinel)
}

const NONE: u32 = u32::MAX;

/// BPE merge on a sequence of initial token IDs.
///
/// Dispatches to a linear-scan or heap-based implementation depending on the
/// number of tokens.  Short pieces benefit from cache-friendly linear scan;
/// long pieces benefit from O(log n) heap-based min-finding.
fn bpe_merge(initial_ids: &[TokenId], ranks: &BpeTable) -> SmallVec<[TokenId; 16]> {
    // Threshold tuned experimentally: BinaryHeap overhead pays off at ~32+ tokens.
    if initial_ids.len() <= 32 {
        bpe_merge_linear(initial_ids, ranks)
    } else {
        bpe_merge_heap(initial_ids, ranks)
    }
}

/// Build the initial linked list from token IDs.
fn build_nodes(ids: &[TokenId]) -> SmallVec<[Node; 32]> {
    let n = ids.len();
    let mut nodes: SmallVec<[Node; 32]> = SmallVec::with_capacity(n);
    for (i, &id) in ids.iter().enumerate() {
        nodes.push(Node {
            token_id: id,
            rank: Rank::MAX,
            prev: if i > 0 { (i - 1) as u32 } else { NONE },
            next: if i + 1 < n { (i + 1) as u32 } else { NONE },
        });
    }
    nodes
}

/// Compute the merge rank for node `i`: rank of merging i with i.next.
#[inline]
fn node_rank(nodes: &[Node], i: usize, ranks: &BpeTable) -> Rank {
    let j = nodes[i].next;
    if j == NONE {
        return Rank::MAX;
    }
    ranks.pair_rank(nodes[i].token_id, nodes[j as usize].token_id)
}

/// Collect surviving token IDs from the linked list.
fn collect_ids(nodes: &[Node]) -> SmallVec<[TokenId; 16]> {
    let mut ids = SmallVec::new();
    let mut cur = 0u32;
    while cur != NONE {
        ids.push(nodes[cur as usize].token_id);
        cur = nodes[cur as usize].next;
    }
    ids
}

/// Linear-scan BPE merge: O(n) per merge step. Best for short pieces.
fn bpe_merge_linear(initial_ids: &[TokenId], ranks: &BpeTable) -> SmallVec<[TokenId; 16]> {
    let n = initial_ids.len();
    debug_assert!(n >= 2);

    let mut nodes = build_nodes(initial_ids);

    // Compute initial pair ranks and find the global minimum.
    let mut min_rank: (Rank, u32) = (Rank::MAX, NONE);
    for i in 0..n.saturating_sub(1) {
        let rank = node_rank(&nodes, i, ranks);
        nodes[i].rank = rank;
        if rank < min_rank.0 {
            min_rank = (rank, i as u32);
        }
    }

    // Merge loop.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1 as usize;
        let j = nodes[i].next as usize;

        // Look up the merged token ID.
        let (_, merged_id) = ranks
            .pair_merge(nodes[i].token_id, nodes[j].token_id)
            .unwrap();
        nodes[i].token_id = merged_id;

        // Unlink j.
        let j_next = nodes[j].next;
        nodes[i].next = j_next;
        if j_next != NONE {
            nodes[j_next as usize].prev = i as u32;
        }

        // Update affected ranks.
        nodes[i].rank = node_rank(&nodes, i, ranks);
        let pred = nodes[i].prev;
        if pred != NONE {
            nodes[pred as usize].rank = node_rank(&nodes, pred as usize, ranks);
        }

        // Linear scan for new minimum.
        min_rank = (Rank::MAX, NONE);
        let mut cur = 0u32;
        loop {
            if nodes[cur as usize].rank < min_rank.0 {
                min_rank = (nodes[cur as usize].rank, cur);
            }
            let next = nodes[cur as usize].next;
            if next == NONE {
                break;
            }
            cur = next;
        }
    }

    collect_ids(&nodes)
}

/// Heap-based BPE merge: O(log n) per merge step. Best for long pieces.
fn bpe_merge_heap(initial_ids: &[TokenId], ranks: &BpeTable) -> SmallVec<[TokenId; 16]> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = initial_ids.len();
    debug_assert!(n >= 2);

    let mut nodes = build_nodes(initial_ids);

    // Priority queue: (rank, node_index). Reverse for min-heap.
    let mut heap: BinaryHeap<Reverse<(Rank, u32)>> = BinaryHeap::with_capacity(n);

    // Compute initial pair ranks.
    for i in 0..n.saturating_sub(1) {
        let rank = node_rank(&nodes, i, ranks);
        nodes[i].rank = rank;
        if rank != Rank::MAX {
            heap.push(Reverse((rank, i as u32)));
        }
    }

    // Merge loop.
    while let Some(Reverse((rank, idx))) = heap.pop() {
        let i = idx as usize;

        // Stale entry: skip if rank no longer matches.
        if nodes[i].rank != rank {
            continue;
        }

        let j = nodes[i].next as usize;

        // Look up the merged token ID.
        let (_, merged_id) = ranks
            .pair_merge(nodes[i].token_id, nodes[j].token_id)
            .unwrap();
        nodes[i].token_id = merged_id;

        // Unlink j.
        let j_next = nodes[j].next;
        nodes[i].next = j_next;
        if j_next != NONE {
            nodes[j_next as usize].prev = i as u32;
        }
        nodes[j].rank = Rank::MAX; // invalidate removed node

        // Update affected ranks and push to heap.
        let new_rank = node_rank(&nodes, i, ranks);
        nodes[i].rank = new_rank;
        if new_rank != Rank::MAX {
            heap.push(Reverse((new_rank, i as u32)));
        }

        let pred = nodes[i].prev;
        if pred != NONE {
            let pred_rank = node_rank(&nodes, pred as usize, ranks);
            nodes[pred as usize].rank = pred_rank;
            if pred_rank != Rank::MAX {
                heap.push(Reverse((pred_rank, pred)));
            }
        }
    }

    collect_ids(&nodes)
}

// ---------------------------------------------------------------------------
// Public API: unified BPE encode
// ---------------------------------------------------------------------------

/// Encode a piece (byte slice + atom-boundary offsets) using BPE.
///
/// `offsets` defines where atoms start: for byte-level BPE, `[0, 1, 2, ..., len]`;
/// for char-level BPE, character boundary offsets from `char_indices()`.
///
/// The merge algorithm works on token-ID pairs, avoiding byte concatenation
/// in the inner loop.
pub fn bpe_encode(
    piece: &[u8],
    offsets: &[usize],
    bpe: &BpeTable,
    byte_fallback: bool,
    unk_token_id: Option<TokenId>,
    out: &mut Vec<TokenId>,
) {
    if piece.is_empty() || offsets.len() < 2 {
        return;
    }

    // Fast path: whole piece is a known token.
    if let Some(id) = bpe.bytes_to_id(piece) {
        out.push(id);
        return;
    }

    let n_atoms = offsets.len() - 1;

    // Single atom, not in vocab → fallback.
    if n_atoms == 1 {
        fallback_into(piece, bpe, byte_fallback, unk_token_id, out);
        return;
    }

    // Two-atom fast path: at most 1 merge possible (already tried whole piece).
    if n_atoms == 2 {
        for w in offsets.windows(2) {
            let span = &piece[w[0]..w[1]];
            match bpe.bytes_to_id(span) {
                Some(id) => out.push(id),
                None => fallback_into(span, bpe, byte_fallback, unk_token_id, out),
            }
        }
        return;
    }

    // Build initial token IDs from atom spans, tracking unresolved atoms.
    let mut initial_ids: SmallVec<[TokenId; 32]> = SmallVec::with_capacity(n_atoms);
    let mut all_resolved = true;
    for w in offsets.windows(2) {
        let span = &piece[w[0]..w[1]];
        match bpe.bytes_to_id(span) {
            Some(id) => initial_ids.push(id),
            None => {
                all_resolved = false;
                initial_ids.push(TokenId::MAX); // sentinel for unresolved
            }
        }
    }

    if all_resolved {
        // All atoms in vocab → run BPE merge on the full sequence.
        let merged = bpe_merge(&initial_ids, bpe);
        out.extend_from_slice(&merged);
        return;
    }

    // Some atoms not in vocab. Segment into contiguous runs of resolved
    // atoms (which get BPE-merged) and unresolved atoms (byte-fallback).
    let mut run_start = 0;
    while run_start < n_atoms {
        if initial_ids[run_start] == TokenId::MAX {
            // Unresolved atom → byte-fallback.
            let span = &piece[offsets[run_start]..offsets[run_start + 1]];
            fallback_into(span, bpe, byte_fallback, unk_token_id, out);
            run_start += 1;
        } else {
            // Find the end of this contiguous resolved run.
            let mut run_end = run_start + 1;
            while run_end < n_atoms && initial_ids[run_end] != TokenId::MAX {
                run_end += 1;
            }
            // BPE-merge this resolved segment.
            let segment = &initial_ids[run_start..run_end];
            if segment.len() == 1 {
                out.push(segment[0]);
            } else {
                let merged = bpe_merge(segment, bpe);
                out.extend_from_slice(&merged);
            }
            run_start = run_end;
        }
    }
}

/// Byte-fallback: encode unknown bytes as `<0xNN>` tokens.
/// Appends directly to `out` — no intermediate allocations.
fn fallback_into(
    bytes: &[u8],
    bpe: &BpeTable,
    byte_fallback: bool,
    unk_token_id: Option<TokenId>,
    out: &mut Vec<TokenId>,
) {
    if byte_fallback {
        for &b in bytes {
            if let Some(id) = bpe.byte_fallback(b).or(unk_token_id) {
                out.push(id);
            }
        }
    } else if let Some(id) = unk_token_id {
        out.push(id);
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers (preserve existing call-site signatures)
// ---------------------------------------------------------------------------

/// Encode a raw byte slice using byte-level BPE (each byte is an atom).
///
/// Used by ByteLevel models (GPT-2, LLaMA, Qwen3).  Offsets are trivially
/// `[0, 1, 2, …, len]` — no `char_indices()` needed.
pub fn bpe_encode_bytes(
    piece: &[u8],
    bpe: &BpeTable,
    byte_fallback: bool,
    unk_token_id: Option<TokenId>,
    out: &mut Vec<TokenId>,
) {
    if piece.is_empty() {
        return;
    }
    let offsets: SmallVec<[usize; 32]> = (0..=piece.len()).collect();
    bpe_encode(piece, &offsets, bpe, byte_fallback, unk_token_id, out);
}

/// Encode a text fragment using char-level BPE (each Unicode char is an atom).
///
/// Used by SentencePiece/Metaspace models (Gemma).  Offsets come from
/// `char_indices()`.
pub fn bpe_encode_chars(
    piece: &str,
    bpe: &BpeTable,
    byte_fallback: bool,
    unk_token_id: Option<TokenId>,
    out: &mut Vec<TokenId>,
) {
    if piece.is_empty() {
        return;
    }
    let offsets: SmallVec<[usize; 32]> = piece
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(piece.len()))
        .collect();
    bpe_encode(piece.as_bytes(), &offsets, bpe, byte_fallback, unk_token_id, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_ranks() -> BpeTable {
        let mut map = HashMap::new();
        map.insert(0u32, b"u".to_vec());
        map.insert(1, b"n".to_vec());
        map.insert(2, b"r".to_vec());
        map.insert(3, b"e".to_vec());
        map.insert(4, b"l".to_vec());
        map.insert(5, b"a".to_vec());
        map.insert(6, b"t".to_vec());
        map.insert(7, b"d".to_vec());
        map.insert(8, b"re".to_vec());
        map.insert(9, b"at".to_vec());
        map.insert(10, b"ed".to_vec());
        map.insert(11, b"un".to_vec());
        map.insert(12, b"ated".to_vec());
        map.insert(13, b"rel".to_vec());
        map.insert(14, b"related".to_vec());
        map.insert(15, b"unrelated".to_vec());
        BpeTable::from_decoder_map(map)
    }

    fn byte_pair_encode(piece: &[u8], bpe: &BpeTable) -> Vec<TokenId> {
        let mut out = Vec::new();
        bpe_encode_bytes(piece, bpe, false, None, &mut out);
        out
    }

    #[test]
    fn test_bpe_merge_levels() {
        let ranks = make_test_ranks();
        assert_eq!(byte_pair_encode(b"unrelated", &ranks), vec![15]); // full
        assert_eq!(byte_pair_encode(b"un", &ranks), vec![11]);        // partial
        assert_eq!(byte_pair_encode(b"u", &ranks), vec![0]);          // atom
        assert!(byte_pair_encode(b"", &ranks).is_empty());            // empty
        assert_eq!(byte_pair_encode(b"unat", &ranks), vec![11, 9]);   // multi-token
    }

    #[test]
    fn test_from_vocab_and_merges() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0u32);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        vocab.insert("ab".to_string(), 3);
        vocab.insert("abc".to_string(), 4);
        vocab.insert("bc".to_string(), 5);

        // a+b has higher priority (rank 1) than b+c (rank 2)
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let bpe = BpeTable::from_vocab_and_merges(&vocab, &merges, "", false);

        let mut out = Vec::new();
        bpe_encode_bytes(b"abc", &bpe, false, None, &mut out);
        assert_eq!(out, vec![4]); // fully merged

        out.clear();
        bpe_encode_bytes(b"ba", &bpe, false, None, &mut out);
        assert_eq!(out, vec![1, 0]); // no merge for b+a
    }

    #[test]
    fn test_byte_fallback() {
        let mut vocab = HashMap::new();
        vocab.insert("h".to_string(), 0u32);
        vocab.insert("i".to_string(), 1);
        vocab.insert("hi".to_string(), 2);
        vocab.insert("<0x80>".to_string(), 10);

        let bpe = BpeTable::from_vocab_and_merges(
            &vocab, &[("h".to_string(), "i".to_string())], "", false,
        );

        let mut out = Vec::new();
        bpe_encode_bytes(b"hi", &bpe, true, None, &mut out);
        assert_eq!(out, vec![2]);

        out.clear();
        bpe_encode_bytes(&[0x80], &bpe, true, None, &mut out);
        assert_eq!(out, vec![10]); // falls back to <0x80>
    }

    #[test]
    fn test_char_level_bpe_multibyte() {
        let mut vocab = HashMap::new();
        vocab.insert("▁".to_string(), 0u32);
        vocab.insert("H".to_string(), 1);
        vocab.insert("i".to_string(), 2);
        vocab.insert("▁H".to_string(), 3);
        vocab.insert("▁Hi".to_string(), 4);

        let merges = vec![
            ("▁".to_string(), "H".to_string()),
            ("▁H".to_string(), "i".to_string()),
        ];
        let bpe = BpeTable::from_vocab_and_merges(&vocab, &merges, "", false);

        let mut out = Vec::new();
        bpe_encode_chars("▁Hi", &bpe, false, None, &mut out);
        assert_eq!(out, vec![4]);
    }

    #[test]
    fn test_gpt2_byte_unicode_roundtrip() {
        let lut = bytes_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for b in 0u16..256 {
            let c = lut[b as usize];
            assert!(c != '\0');
            assert!(seen.insert(c), "duplicate mapping for byte {b:#x}");
            assert_eq!(CHAR_TO_BYTE[c as usize], Some(b as u8));
        }
    }
}

