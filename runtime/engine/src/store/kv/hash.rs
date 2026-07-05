//! Pure semantic-hash functions for the KV store (kv_refact.md,
//! "Token-Slot Hashes, Page Hashes, and Trie Matching").
//!
//! Three granularities, all `[u8; 32]`:
//! - token-slot hash: identity of one committed token slot. Chained over the
//!   cache domain and all causal inputs, so equal hashes imply equal content
//!   *and* equal causal history.
//! - page hash: deterministic fold of one page's format/validity state and its
//!   ordered token-slot hashes. Needs only that page's slot hashes.
//! - path hash: fold of all visible page hashes from the trie root through a
//!   node, one page at a time, so it is independent of where radix node
//!   boundaries fall.
//!
//! These are semantic fingerprints, not hashes of KV bytes. No mutable state
//! lives here; results are stored in trie node vectors by `KvPageTable`.

/// A 256-bit semantic hash.
pub type Hash256 = [u8; 32];

const DOMAIN_CONTEXT: &str = "pie kv cache domain v1";
const SLOT_CONTEXT: &str = "pie kv token-slot hash v1";
const OPAQUE_CONTEXT: &str = "pie kv opaque token-slot hash v1";
const PAGE_CONTEXT: &str = "pie kv page hash v1";
const PATH_CONTEXT: &str = "pie kv path hash v1";

/// The pass-wide cache domain folded into every token-slot hash. `seed`
/// scopes the domain: today the per-store boot nonce (hashes never cross a
/// store or a boot); a model/weights-stable seed replaces it when caches
/// persist across runs.
pub fn cache_domain(seed: &[u8]) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(DOMAIN_CONTEXT);
    hasher.update(seed);
    *hasher.finalize().as_bytes()
}

/// Recipe-derived token-slot hash: chains the cache domain (model/weights,
/// adapters, KV format, and other pass-wide inputs, pre-folded into `domain`),
/// the previous slot's hash (`None` for the first slot), and this slot's
/// causal inputs. Standard causal text rows hash `(token, position)`.
pub fn chain_token_slot_hash(
    domain: &Hash256,
    prev: Option<&Hash256>,
    token: u32,
    position: u32,
) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(SLOT_CONTEXT);
    hasher.update(domain);
    match prev {
        Some(prev) => {
            hasher.update(&[1]);
            hasher.update(prev);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&token.to_le_bytes());
    hasher.update(&position.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// Fresh opaque token-slot hash for slots no recipe covers. Preserves concrete
/// identity across forks and parent selections but cannot be reproduced by an
/// unrelated forward. `nonce` is a per-store random value; `counter` must be
/// unique per issued hash within that store.
pub fn opaque_token_slot_hash(nonce: &Hash256, counter: u64) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(OPAQUE_CONTEXT);
    hasher.update(nonce);
    hasher.update(&counter.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// Page hash: folds the page's slot count, per-slot validity, and the ordered
/// token-slot hashes. `None` slots (unwritten/invalid) are folded as absent so
/// partial pages hash differently from complete ones.
pub fn page_hash(slot_hashes: &[Option<Hash256>]) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(PAGE_CONTEXT);
    hasher.update(&(slot_hashes.len() as u64).to_le_bytes());
    for slot in slot_hashes {
        match slot {
            Some(hash) => {
                hasher.update(&[1]);
                hasher.update(hash);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    *hasher.finalize().as_bytes()
}

/// Fold `page_hashes` onto a running path hash, one page at a time. Returns
/// `prev` unchanged when `page_hashes` is empty; `None` only for an empty
/// path. Per-page chaining makes the result independent of node boundaries.
pub fn fold_path_hash(prev: Option<Hash256>, page_hashes: &[Hash256]) -> Option<Hash256> {
    let mut acc = prev;
    for page in page_hashes {
        let mut hasher = blake3::Hasher::new_derive_key(PATH_CONTEXT);
        match &acc {
            Some(acc) => {
                hasher.update(&[1]);
                hasher.update(acc);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(page);
        acc = Some(*hasher.finalize().as_bytes());
    }
    acc
}
