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
