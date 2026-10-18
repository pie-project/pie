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
}
