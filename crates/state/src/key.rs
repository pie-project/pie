//! Content keys.
//!
//! A token slot's hash chains the one before it, its token and its position,
//! under a domain derived from the model's identity: it names the whole
//! prefix up to that slot. A page's key hashes its slots', so it names the
//! whole prefix up to the page's end too, the same on every worker of one
//! model and page size.

pub type Hash256 = [u8; 32];

const DOMAIN_CONTEXT: &str = "pie kv cache domain v1";
const SLOT_CONTEXT: &str = "pie kv token-slot hash v1";
const PAGE_CONTEXT: &str = "pie kv page hash v1";

pub fn cache_domain(seed: &[u8]) -> Hash256 {
    let mut hasher = blake3::Hasher::new_derive_key(DOMAIN_CONTEXT);
    hasher.update(seed);
    *hasher.finalize().as_bytes()
}

pub fn chain_token_slot_hash(
    domain: &Hash256,
    prev: Option<&Hash256>,
    token: u32,
    position: u32,
) -> Hash256 {
    // The same bytes as one update each, into the context's key derived once.
    let mut bytes = [0u8; 32 + 1 + 32 + 4 + 4];
    bytes[..32].copy_from_slice(domain);
    let mut at = 32;
    match prev {
        Some(prev) => {
            bytes[at] = 1;
            bytes[at + 1..at + 33].copy_from_slice(prev);
            at += 33;
        }
        None => {
            bytes[at] = 0;
            at += 1;
        }
    }
    bytes[at..at + 4].copy_from_slice(&token.to_le_bytes());
    bytes[at + 4..at + 8].copy_from_slice(&position.to_le_bytes());
    let mut hasher = derived(SLOT_CONTEXT).clone();
    hasher.update(&bytes[..at + 8]);
    *hasher.finalize().as_bytes()
}

/// `blake3::Hasher::new_derive_key(context)`, derived once a context.
fn derived(context: &'static str) -> &'static blake3::Hasher {
    use std::sync::OnceLock;
    static SLOT: OnceLock<blake3::Hasher> = OnceLock::new();
    static PAGE: OnceLock<blake3::Hasher> = OnceLock::new();
    let cell = if context == SLOT_CONTEXT {
        &SLOT
    } else {
        &PAGE
    };
    debug_assert!(context == SLOT_CONTEXT || context == PAGE_CONTEXT);
    cell.get_or_init(|| blake3::Hasher::new_derive_key(context))
}

pub fn chain_slots(
    domain: &Hash256,
    mut prev: Option<Hash256>,
    start: u64,
    tokens: &[u32],
) -> Vec<Hash256> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, &token)| {
            let hash =
                chain_token_slot_hash(domain, prev.as_ref(), token, (start + i as u64) as u32);
            prev = Some(hash);
            hash
        })
        .collect()
}

/// A page's hash over its slots: what one engine's cache keys a page by. It
/// depends on the page size.
pub fn page_hash(slot_hashes: &[Option<Hash256>]) -> Hash256 {
    let mut bytes = Vec::with_capacity(8 + slot_hashes.len() * 33);
    bytes.extend_from_slice(&(slot_hashes.len() as u64).to_le_bytes());
    for slot in slot_hashes {
        match slot {
            Some(hash) => {
                bytes.push(1);
                bytes.extend_from_slice(hash);
            }
            None => bytes.push(0),
        }
    }
    let mut hasher = derived(PAGE_CONTEXT).clone();
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_names_its_whole_prefix() {
        let domain = cache_domain(b"model");
        let a = chain_slots(&domain, None, 0, &[1, 2, 3]);
        let b = chain_slots(&domain, None, 0, &[9, 2, 3]);
        assert_ne!(a[2], b[2], "an earlier token changes every later key");
        let other = chain_slots(&cache_domain(b"other"), None, 0, &[1, 2, 3]);
        assert_ne!(a[2], other[2], "another model's domain changes every key");
    }

    #[test]
    fn the_keys_are_the_ones_hashed_piece_by_piece() {
        let slot = |domain: &Hash256, prev: Option<&Hash256>, token: u32, position: u32| {
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
        };
        let page = |slots: &[Option<Hash256>]| {
            let mut hasher = blake3::Hasher::new_derive_key(PAGE_CONTEXT);
            hasher.update(&(slots.len() as u64).to_le_bytes());
            for slot in slots {
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
        };
        let domain = cache_domain(b"model");
        let slots = chain_slots(&domain, None, 7, &[5, 0, u32::MAX, 42]);
        assert_eq!(slots[0], slot(&domain, None, 5, 7));
        assert_eq!(slots[3], slot(&domain, Some(&slots[2]), 42, 10));
        let mixed = [Some(slots[0]), None, Some(slots[2]), Some(slots[3])];
        assert_eq!(page_hash(&mixed), page(&mixed));
        assert_eq!(page_hash(&[]), page(&[]));
    }
}
