pub const ROLES: &[&str] = &[
    "norm",
    "gemm",
    "dist",
    "moe",
    "gate",
    "layout",
    "ssm",
    "mla",
    "index",
    "pool",
    "hc",
    "rmsnorm",
    // `attention` carries the whole core family now: `attention.decode`,
    // `.prefill`, `.masked`, the two `_lse` readings, `.sink`,
    // `.merge_lse`, `.logit_softcap`, `.kv_append` and `.kv_append_shared`
    // all read their role off this one entry. `lse_ln`, `logit_softcap`
    // and `res_blend` retired as BARE ROLES with the migration — each was
    // a role for a single point, and each point wears its family's prefix
    // now (`res_blend` joined `norm`). `attention.lse_ln` then retired as
    // a point too, when the floor stated the base of an lse.
    "attention",
    // `kv_append` keeps its bare prefix: the shader planes' paged append
    // claims it, and dsv4's shared plane states `kv_append.shared`. The
    // three appends that MOVED are the ones whose caches have owners —
    // `kv_append.{mla,index,pool}` are `{mla,index,pool}.kv_append` now.
    "kv_append",
    "rope",
    "mlp",
];

#[must_use]
pub const fn is_role(claim: &str) -> bool {
    let bytes = claim.as_bytes();
    let mut role_len = bytes.len();
    let mut k = 0;
    while k < bytes.len() {
        if bytes[k] == b'.' {
            role_len = k;
            break;
        }
        k += 1;
    }
    let mut i = 0;
    while i < ROLES.len() {
        let (a, b) = (ROLES[i].as_bytes(), claim.as_bytes());
        if a.len() == role_len {
            let mut j = 0;
            let mut eq = true;
            while j < a.len() {
                if a[j] != b[j] {
                    eq = false;
                    break;
                }
                j += 1;
            }
            if eq {
                return true;
            }
        }
        i += 1;
    }
    false
}
