#!/bin/bash
# Mutation-tests the golden in tests/kv_cache_parity.rs.
#
# A golden proves the two implementations agree. It does not prove the sweep
# would NOTICE a disagreement -- a transcript that omits the interesting
# columns passes just as happily when the port is wrong. Each mutation below
# is a plausible porting slip applied to the Rust; the golden must reject it.
#
# The final entry is a NO-OP control. It must MISS. If it is ever reported as
# caught, the harness is detecting its own edit rather than the behaviour, and
# every other result here is meaningless.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(cd "$HERE/../../.." && pwd)"
SRC="$CRATE/src/store/kv_cache.rs"
GEO="$CRATE/src/store/kv_geometry.rs"
BACKUP="$(mktemp)"
BACKUP_GEO="$(mktemp)"
cp "$SRC" "$BACKUP"
cp "$GEO" "$BACKUP_GEO"
restore() { cp "$BACKUP" "$SRC"; cp "$BACKUP_GEO" "$GEO"; rm -f "$BACKUP" "$BACKUP_GEO"; }
trap restore EXIT

pass=0
fail=0

# $1 label, $2 expectation (catch|miss), $3 file, $4 from, $5 to
mutate() {
    local label="$1" expect="$2" file="$3" from="$4" to="$5"
    cp "$BACKUP" "$SRC"; cp "$BACKUP_GEO" "$GEO"
    local target="$SRC"
    [[ "$file" == "geo" ]] && target="$GEO"
    if ! grep -qF -- "$from" "$target"; then
        echo "  SKIP  $label (pattern not found -- the code moved)"
        fail=$((fail + 1))
        return
    fi
    python3 - "$target" "$from" "$to" <<'PY'
import sys
path, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
open(path, "w", encoding="utf-8").write(s.replace(a, b, 1))
PY
    local got="catch"
    if (cd "$CRATE" && cargo test --features cuda-13 --test kv_cache_parity \
            >/dev/null 2>&1); then
        got="miss"
    fi
    if [[ "$got" == "$expect" ]]; then
        echo "  ok    $label ($got)"
        pass=$((pass + 1))
    else
        echo "  FAIL  $label (expected $expect, got $got)"
        fail=$((fail + 1))
    fi
}

echo "mutating $SRC and $GEO ..."

# --- the storage tier -------------------------------------------------------
mutate "storage uses the logical head_dim, not the packed one" catch src \
    'PageOrder::Nhd => vec![pages, psz, heads, storage_hd],' \
    'PageOrder::Nhd => vec![pages, psz, heads, logical_hd],'
mutate "storage extents transposed to HND" catch src \
    'PageOrder::Nhd => vec![pages, psz, heads, storage_hd],' \
    'PageOrder::Nhd => vec![pages, heads, psz, storage_hd],'
mutate "K and V swap dtype" catch src \
    'v: Some(TensorSpec::new(dt, storage_shape)?),' \
    'v: Some(TensorSpec::new(DType::Bf16, storage_shape)?),'

# --- the scale tier ---------------------------------------------------------
mutate "PerTokenHead scale gains a trailing head_dim" catch src \
    'let s = TensorSpec::new(DType::Fp32, vec![pages, psz, heads])?;' \
    'let s = TensorSpec::new(DType::Fp32, vec![pages, psz, heads, logical_hd])?;'
mutate "scale tier is fp16 rather than fp32" catch src \
    'let s = TensorSpec::new(DType::Fp32, vec![pages, psz, heads])?;' \
    'let s = TensorSpec::new(DType::Fp16, vec![pages, psz, heads])?;'
mutate "block count floors instead of ceiling" catch src \
    'let blocks = (logical_hd + bs - 1) / bs;' \
    'let blocks = logical_hd / bs;'
mutate "block_size 0 falls back to 1 rather than 16" catch src \
    '                } else {
                    16
                };' \
    '                } else {
                    1
                };'
mutate "blocked scale sized on the packed head_dim" catch src \
    'let blocks = (logical_hd + bs - 1) / bs;' \
    'let blocks = (storage_hd + bs - 1) / bs;'

# --- the dequantisation mirror ---------------------------------------------
mutate "mirror allocated for native bf16 too" catch src \
    'if !self.format.is_native_bf16() {
            let m = TensorSpec::new' \
    'if true {
            let m = TensorSpec::new'
mutate "mirror sized on the packed head_dim" catch src \
    'let m = TensorSpec::new(DType::Bf16, vec![pages, psz, heads, logical_hd])?;' \
    'let m = TensorSpec::new(DType::Bf16, vec![pages, psz, heads, storage_hd])?;'
mutate "mirror keeps the storage dtype" catch src \
    'let m = TensorSpec::new(DType::Bf16, vec![pages, psz, heads, logical_hd])?;' \
    'let m = TensorSpec::new(dt, vec![pages, psz, heads, logical_hd])?;'

# --- aliasing ---------------------------------------------------------------
mutate "an aliased layer allocates anyway" catch src \
    'if !self.owns_pages(layer) {
            return Ok(LayerSlot::default());
        }' \
    'if false {
            return Ok(LayerSlot::default());
        }'
# Expected to MISS, and that is a finding about my own port rather than a hole
# in the sweep: `resolve` already falls back to `layer` when the vector is
# empty, so the `is_empty()` arm in `owns_pages` is unreachable-by-value. It is
# kept because it mirrors the C++ shape, but it cannot change an answer.
mutate "empty kv_source_layer arm is redundant with resolve's fallback" miss src \
    'self.per_layer.kv_source_layer.is_empty() || self.resolve(layer) == layer' \
    'self.resolve(layer) == layer'
mutate "resolve falls back to 0 rather than the layer" catch src \
    '            .copied()
            .unwrap_or(layer)
    }

    fn owns_pages' \
    '            .copied()
            .unwrap_or(0)
    }

    fn owns_pages'

# --- the per-layer fallbacks ------------------------------------------------
mutate "head_dim_at falls back to the first entry" catch src \
    '            .copied()
            .unwrap_or(self.head_dim)' \
    '            .copied()
            .unwrap_or_else(|| self.per_layer.head_dim.first().copied().unwrap_or(0))'
mutate "num_kv_heads_at falls back to head_dim" catch src \
    '            .copied()
            .unwrap_or(self.num_kv_heads)' \
    '            .copied()
            .unwrap_or(self.head_dim)'
mutate "scalar head_dim defaults to the argument, not zero" catch src \
    'let head_dim = per_layer.head_dim.first().copied().unwrap_or(0);' \
    'let head_dim = per_layer.head_dim.first().copied().unwrap_or(num_kv_heads);'

# --- the length validations -------------------------------------------------
mutate "length check accepts a longer vector" catch src \
    'if !v.is_empty() && i32::try_from(v.len()).unwrap_or(i32::MAX) != num_layers {' \
    'if !v.is_empty() && i32::try_from(v.len()).unwrap_or(i32::MAX) < num_layers {'
mutate "length check skipped for kv_source_layer" catch src \
    '("kv_source_layer", &per_layer.kv_source_layer),' \
    '("kv_source_layer", &Vec::new()),'
mutate "validation order: kv heads reported before head dim" catch src \
    '            ("per_layer_head_dim", &per_layer.head_dim),
            ("kv_source_layer", &per_layer.kv_source_layer),' \
    '            ("kv_source_layer", &per_layer.kv_source_layer),
            ("per_layer_head_dim", &per_layer.head_dim),'

# --- envelopes --------------------------------------------------------------
mutate "envelopes allocated for a non-native format" catch src \
    'if !self.format.is_native_bf16() || self.page_order() == PageOrder::Hnd {
            return Ok(());
        }' \
    'if false {
            return Ok(());
        }'
mutate "envelope keeps the token extent" catch src \
    'vec![i64::from(self.num_pages), i64::from(kvh), i64::from(hd)],' \
    'vec![i64::from(self.num_pages), i64::from(self.page_size), i64::from(kvh), i64::from(hd)],'
mutate "envelope skips on is_alias rather than the null pointer" catch src \
    'if !self.slots[idx].has_key_pointer() {' \
    'if self.slots[idx].is_alias() {'
mutate "envelope dtype is fp32" catch src \
    '            let e = TensorSpec::new(
                DType::Bf16,' \
    '            let e = TensorSpec::new(
                DType::Fp32,'
mutate "envelopes_enabled set even when the tier is skipped" catch src \
    'if !self.format.is_native_bf16() || self.page_order() == PageOrder::Hnd {
            return Ok(());' \
    'if !self.format.is_native_bf16() || self.page_order() == PageOrder::Hnd {
            self.envelopes = true;
            return Ok(());'
mutate "env switch accepts any non-empty value" catch src \
    'Ok(v) => v == "1" || v == "true" || v == "on",' \
    'Ok(v) => !v.is_empty(),'
mutate "env switch is case-insensitive" catch src \
    'Ok(v) => v == "1" || v == "true" || v == "on",' \
    'Ok(v) => { let v = v.to_lowercase(); v == "1" || v == "true" || v == "on" }'

# --- page_buffers -----------------------------------------------------------
mutate "page_buffers does not resolve through an alias" catch src \
    'let src = self.resolve(layer);
        let hd = u32::try_from(self.head_dim_at(src)' \
    'let src = layer;
        let hd = u32::try_from(self.head_dim_at(src)'
mutate "page_buffers emits scales unconditionally" catch src \
    'if scale > 0 {' \
    'if true {'

# --- the free functions the planner shares ----------------------------------
mutate "per-page bytes omit the dequant scratch" catch geo \
    'if !format.is_native_bf16() {
        bytes += 2' \
    'if false {
        bytes += 2'
mutate "dequant scratch sized on the packed head_dim" catch geo \
    'bytes += 2 * u64::from(page_size)
            * u64::from(num_kv_heads)
            * u64::from(head_dim)' \
    'bytes += 2 * u64::from(page_size)
            * u64::from(num_kv_heads)
            * u64::from(format.storage_head_dim(head_dim))'

# --- the control ------------------------------------------------------------
# The control must still COMPILE -- a mutation that fails to build is reported
# as caught by any harness at all, which proves nothing. This one is an
# algebraically identical rewrite of the ceiling division.
mutate "CONTROL: reparenthesise the ceiling division (no-op)" miss src \
    'let blocks = (logical_hd + bs - 1) / bs;' \
    'let blocks = (logical_hd + (bs - 1)) / bs;'

echo
echo "mutations: $pass as expected, $fail unexpected"
[[ "$fail" -eq 0 ]]
