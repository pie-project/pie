use inferlet::eta::{Pipeline, RsWorkingSet, WorkingSet};

const SALT_A: u64 = 0x9E37_79B9_7F4A_7C15;
const SALT_B: u64 = 0xC2B2_AE3D_27D4_EB4F;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct Hit {
    pub tokens: u32,
    pub kv: WorkingSet,
    pub rs: Option<RsWorkingSet>,
}

pub struct Prefix {
    page_t: u32,
    keys: Vec<Vec<u8>>,
}

impl Prefix {
    pub fn new(prompt: &[u32], page_t: u32) -> Self {
        let pages = (prompt.len().saturating_sub(1) / page_t as usize) as u32;
        let mut keys = Vec::with_capacity(pages as usize);
        let (mut a, mut b) = (SALT_A, SALT_B);
        for (i, &token) in prompt[..(pages * page_t) as usize].iter().enumerate() {
            let x = u64::from(token) | ((i as u64) << 32);
            a = mix(a ^ x).wrapping_add(SALT_B);
            b = mix(b.wrapping_add(x.rotate_left(21)) ^ SALT_A);
            if (i as u32 + 1) % page_t == 0 {
                let mut key = Vec::with_capacity(24);
                key.extend_from_slice(b"pc1.");
                key.extend_from_slice(&a.to_le_bytes());
                key.extend_from_slice(&b.to_le_bytes());
                key.extend_from_slice(&(i as u32 + 1).to_le_bytes());
                keys.push(key);
            }
        }
        Self { page_t, keys }
    }

    pub fn find(&self, hybrid: bool) -> Option<Hit> {
        let mut candidates: Vec<Vec<u8>> = self.keys.iter().rev().cloned().collect();
        let total = candidates.len();
        let mut skipped = 0usize;
        while !candidates.is_empty() {
            let (position, kv) = WorkingSet::find_index(&candidates).ok()??;
            let absolute = skipped + position;
            let pages = (total - absolute) as u32;
            if !hybrid {
                return Some(Hit {
                    tokens: pages * self.page_t,
                    kv,
                    rs: None,
                });
            }
            if let Ok(Some(rs)) = RsWorkingSet::from_index(&candidates[position]) {
                return Some(Hit {
                    tokens: pages * self.page_t,
                    kv,
                    rs: Some(rs),
                });
            }
            candidates.drain(..=position);
            skipped = absolute + 1;
        }
        None
    }

    pub fn publish(&self, pages: u32, kv: &WorkingSet, rs: &[RsWorkingSet], pipe: &Pipeline) {
        let Some(key) = pages.checked_sub(1).and_then(|i| self.keys.get(i as usize)) else {
            return;
        };
        let Ok(prefix) = kv.slice(pipe, 0, pages) else {
            return;
        };
        if prefix.update_index(key).is_err() {
            return;
        }
        if let Some(state) = rs.first()
            && let Ok(snapshot) = state.fork(pipe)
        {
            let _ = snapshot.update_index(key);
        }
    }
}

pub fn spans(base: u32, n: u32, page_t: u32, cap: u32) -> Vec<(u32, u32)> {
    let aligned = (n.saturating_sub(1) / page_t) * page_t;
    let step = (cap / page_t).max(1) * page_t;
    let mut out = Vec::new();
    let mut at = base;
    while at < aligned {
        let end = (at + step).min(aligned);
        out.push((at, end));
        at = end;
    }
    out.push((at, n));
    out
}
