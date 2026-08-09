#!/bin/bash
# Mutation-tests the golden in tests/caches_parity.rs.
#
# A golden proves the two implementations agree; it does not prove the sweep
# would NOTICE a disagreement. Each mutation below is a plausible porting slip
# applied to the Rust. The golden must reject every one of them.
#
# The final entry is a NO-OP control that must still COMPILE and must MISS. A
# mutation that fails to build registers as caught by any harness at all, so a
# non-compiling control proves nothing about this one -- which is why the
# control is an algebraically identical rewrite and not a rename.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(cd "$HERE/../../.." && pwd)"
MLA="$CRATE/src/store/mla_cache.rs"
DSV4="$CRATE/src/store/dsv4_compress_cache.rs"
POOL="$CRATE/src/store/swap_pool.rs"
B1="$(mktemp)"; B2="$(mktemp)"; B3="$(mktemp)"
cp "$MLA" "$B1"; cp "$DSV4" "$B2"; cp "$POOL" "$B3"
restore() { cp "$B1" "$MLA"; cp "$B2" "$DSV4"; cp "$B3" "$POOL"; rm -f "$B1" "$B2" "$B3"; }
trap restore EXIT

pass=0
fail=0

# $1 label, $2 expectation (catch|miss), $3 file (mla|dsv4|pool), $4 from, $5 to
mutate() {
    local label="$1" expect="$2" file="$3" from="$4" to="$5"
    cp "$B1" "$MLA"; cp "$B2" "$DSV4"; cp "$B3" "$POOL"
    local target="$MLA"
    [[ "$file" == "dsv4" ]] && target="$DSV4"
    [[ "$file" == "pool" ]] && target="$POOL"
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
    if (cd "$CRATE" && cargo test --features cuda-13 --test caches_parity \
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

echo "mutating the three cache layouts ..."

# --- MlaCache: validation ---------------------------------------------------
mutate "mla: page_size may be zero" catch mla \
    '            || page_size <= 0' '            || page_size < 0'
mutate "mla: kv_lora_rank may be zero" catch mla \
    '            || kv_lora_rank <= 0' '            || kv_lora_rank < 0'
mutate "mla: qk_rope_head_dim is not checked" catch mla \
    '            || qk_rope_head_dim <= 0' '            || qk_rope_head_dim < i32::MIN'
mutate "mla: num_layers may be zero" catch mla \
    '        if num_layers <= 0' '        if num_layers < 0'
mutate "mla: num_pages may be zero" catch mla \
    '            || num_pages <= 0' '            || num_pages < 0'
mutate "mla: fp32 storage is accepted" catch mla \
    '        if dtype != DType::Bf16 && dtype != DType::Fp16 {' \
    '        if dtype != DType::Bf16 && dtype != DType::Fp16 && dtype != DType::Fp32 {'
mutate "mla: fp16 storage is refused" catch mla \
    '        if dtype != DType::Bf16 && dtype != DType::Fp16 {' \
    '        if dtype != DType::Bf16 {'
mutate "mla: dtype is checked before the dimensions" catch mla \
    '        if num_layers <= 0' '        if dtype != DType::Bf16 && dtype != DType::Fp16 {
            return Err(Error::invalid(
                "mla_cache",
                "only bf16/fp16 storage is supported",
            ));
        }
        if num_layers <= 0'

# --- MlaCache: geometry -----------------------------------------------------
mutate "mla: ckv and kpe swap their widths" catch mla \
    '            ckv: shape(kv_lora_rank)?,
            kpe: shape(qk_rope_head_dim)?,' \
    '            ckv: shape(qk_rope_head_dim)?,
            kpe: shape(kv_lora_rank)?,'
mutate "mla: the page axis is dropped from the tensor" catch mla \
    '                vec![i64::from(num_pages), i64::from(page_size), i64::from(last)],' \
    '                vec![i64::from(num_pages) * i64::from(page_size), i64::from(last)],'
mutate "mla: allocation groups ckv before kpe" catch mla \
    '            .flat_map(|l| [(l, "ckv", &self.ckv), (l, "kpe", &self.kpe)])
            .collect()' \
    '            .map(|l| (l, "ckv", &self.ckv))
            .chain((0..self.num_layers).map(|l| (l, "kpe", &self.kpe)))
            .collect()'
mutate "mla: a page buffer spans the whole tensor" catch mla \
    '        let per = u64::from(self.page_size) * elem;' \
    '        let per = u64::from(self.page_size) * u64::from(self.num_pages) * elem;'
mutate "mla: both page buffers use the latent width" catch mla \
    '                page_bytes: per * u64::from(self.qk_rope_head_dim),' \
    '                page_bytes: per * u64::from(self.kv_lora_rank),'
mutate "mla: the view reports the argument order it was built from" catch mla \
    '            kv_lora_rank: self.kv_lora_rank,
            qk_rope_head_dim: self.qk_rope_head_dim,' \
    '            kv_lora_rank: self.qk_rope_head_dim,
            qk_rope_head_dim: self.kv_lora_rank,'
mutate "mla: the view reports pages where page_size belongs" catch mla \
    '            num_pages: self.num_pages,
            page_size: self.page_size,' \
    '            num_pages: self.page_size,
            page_size: self.num_pages,'

# --- DsV4CompressCache: which layers allocate -------------------------------
mutate "dsv4: ratio 0 allocates too" catch dsv4 \
    '            if ratio <= 0 {' '            if ratio < 0 {'
mutate "dsv4: a negative ratio allocates" catch dsv4 \
    '            if ratio <= 0 {' '            if false {'
mutate "dsv4: a missing ratio defaults to compressing" catch dsv4 \
    '            let ratio = ratios.get(li).copied().unwrap_or(0);' \
    '            let ratio = ratios.get(li).copied().unwrap_or(1);'
mutate "dsv4: the table is sized from the ratios, not the layers" catch dsv4 \
    '        for li in 0..num_hidden_layers as usize {' \
    '        for li in 0..ratios.len() {'
mutate "dsv4: an empty ratios list still builds a cache" catch dsv4 \
    '        if ratios.is_empty() || num_pages <= 0 || page_size <= 0 {' \
    '        if num_pages <= 0 || page_size <= 0 {'
mutate "dsv4: zero pages still builds a cache" catch dsv4 \
    '        if ratios.is_empty() || num_pages <= 0 || page_size <= 0 {' \
    '        if ratios.is_empty() || num_pages < 0 || page_size <= 0 {'
mutate "dsv4: zero page_size still builds a cache" catch dsv4 \
    '        if ratios.is_empty() || num_pages <= 0 || page_size <= 0 {' \
    '        if ratios.is_empty() || num_pages <= 0 || page_size < 0 {'
mutate "dsv4: a negative layer count is clamped instead of refused" catch dsv4 \
    '        if num_hidden_layers < 0 {' '        if false {'

# --- DsV4CompressCache: widths ----------------------------------------------
mutate "dsv4: ratio 4 gets no wide window" catch dsv4 \
    '            let width = compressor_coff(ratio) as i32 * head_dim;' \
    '            let width = head_dim;'
mutate "dsv4: comp_kv is as wide as the state" catch dsv4 \
    '                comp_kv: spec(head_dim)?,' '                comp_kv: spec(width)?,'
mutate "dsv4: state_score is a comp_kv-width tensor" catch dsv4 \
    '                state_score: spec(width)?,' '                state_score: spec(head_dim)?,'
mutate "dsv4: the storage dtype is fp32" catch dsv4 \
    '                    DType::Bf16,
                    vec![i64::from(num_pages), i64::from(page_size), i64::from(w)],' \
    '                    DType::Fp32,
                    vec![i64::from(num_pages), i64::from(page_size), i64::from(w)],'
mutate "dsv4: a negative extent is clamped instead of refused" catch dsv4 \
    '                    vec![i64::from(num_pages), i64::from(page_size), i64::from(w)],' \
    '                    vec![i64::from(num_pages), i64::from(page_size), i64::from(w.max(0))],'

# --- DsV4CompressCache: the zeroing pass ------------------------------------
mutate "dsv4: a failed memset abandons the whole cache" catch dsv4 \
    '                if !memset(li, name, spec.nbytes()) {
                    break;
                }' \
    '                if !memset(li, name, spec.nbytes()) {
                    return;
                }'
mutate "dsv4: a failed memset only skips one tensor" catch dsv4 \
    '                if !memset(li, name, spec.nbytes()) {
                    break;
                }' \
    '                if !memset(li, name, spec.nbytes()) {
                    continue;
                }'
mutate "dsv4: zero-byte tensors are memset too" catch dsv4 \
    '                if spec.nbytes() == 0 {
                    continue;
                }' \
    '                if false {
                    continue;
                }'
mutate "dsv4: the zeroing order is comp_kv first" catch dsv4 \
    '            ("state_kv", &self.state_kv),
            ("state_score", &self.state_score),
            ("comp_kv", &self.comp_kv),' \
    '            ("comp_kv", &self.comp_kv),
            ("state_kv", &self.state_kv),
            ("state_score", &self.state_score),'

# --- DsV4CompressCache: the accessors ---------------------------------------
mutate "dsv4: has_layer ignores whether the tensor has bytes" catch dsv4 \
    '        self.layer(li).is_some_and(|l| l.state_kv.nbytes() > 0)' \
    '        self.layer(li).is_some()'
mutate "dsv4: is_empty asks whether anything was allocated" catch dsv4 \
    '        self.layers.is_empty()' \
    '        self.layers.iter().all(Option::is_none)'
mutate "dsv4: the page size survives a rejected geometry" catch dsv4 \
    '            return Ok(Self::default());' \
    '            return Ok(Self { page_size, layers: Vec::new() });'

# --- SwapPool: the early return ---------------------------------------------
mutate "pool: bytes_per_page is computed after the early return" catch pool \
    '        if num_pages <= 0 || num_layers <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {' \
    '        if num_pages <= 0 || num_layers <= 0 {
            out.bytes_per_page = 0;
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {'
mutate "pool: the cache constructor keeps bytes_per_page on the early path" catch pool \
    '        if num_pages <= 0 || num_layers <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for (layer, widths) in device_buffers.iter().enumerate() {' \
    '        if num_pages <= 0 || num_layers <= 0 {
            out.bytes_per_page = device_buffers.iter().flatten().sum();
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for (layer, widths) in device_buffers.iter().enumerate() {'
mutate "pool: zero pages still builds the pool" catch pool \
    '        if num_pages <= 0 || num_layers <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {' \
    '        if num_pages < 0 || num_layers <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {'
mutate "pool: a zero page_size gates the early return too" catch pool \
    '        if num_pages <= 0 || num_layers <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {' \
    '        if num_pages <= 0 || num_layers <= 0 || page_size <= 0 {
            return out;
        }
        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {'

# --- SwapPool: the arithmetic -----------------------------------------------
mutate "pool: negative dimensions clamp instead of wrapping" catch pool \
    '    v as i64 as u64' '    v.max(0) as u64'
mutate "pool: bytes_per_page counts one buffer per layer" catch pool \
    '        let bytes_per_page = 2u64
            .wrapping_mul(as_size_t(num_layers))' \
    '        let bytes_per_page = 1u64
            .wrapping_mul(as_size_t(num_layers))'
mutate "pool: bytes_per_page is per layer, not per stack" catch pool \
    '        let bytes_per_page = 2u64
            .wrapping_mul(as_size_t(num_layers))
            .wrapping_mul(one_page);' \
    '        let bytes_per_page = 2u64.wrapping_mul(one_page);'
mutate "pool: the dtype does not scale the page" catch pool \
    '            .wrapping_mul(dtype.size_bytes() as u64);' \
    '            .wrapping_mul(1);'
mutate "pool: a host region holds one page" catch pool \
    '                    nbytes: one_page.wrapping_mul(np),' \
    '                    nbytes: one_page,'
mutate "pool: three host buffers per layer" catch pool \
    '            for buffer in 0..2 {' '            for buffer in 0..3 {'
mutate "pool: the cache pool sizes every buffer from the first" catch pool \
    '                out.bytes_per_page = out.bytes_per_page.wrapping_add(page_bytes);' \
    '                out.bytes_per_page = page_bytes;'
mutate "pool: the cache pool allocates one page per region" catch pool \
    '                    nbytes: page_bytes.wrapping_mul(np),' \
    '                    nbytes: page_bytes,'
mutate "pool: only the restore stream is created" catch pool \
    '        out.streams = StreamPlan { evict: true, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {' \
    '        out.streams = StreamPlan { evict: false, restore: true };
        let np = as_size_t(num_pages);
        for layer in 0..num_layers as u32 {'

# --- expected misses --------------------------------------------------------
#
# `check_against` is a check the C++ does not have, so no oracle row can
# depend on it. It is covered by the unit tests in swap_pool.rs instead. Listed
# here so that a future reader does not mistake its absence for an oversight.
mutate "pool: check_against never rejects" miss pool \
    '            if host.len() != widths.len() {' '            if false {'

# --- the control ------------------------------------------------------------
#
# Algebraically identical and still compiles. A control that fails to BUILD
# would register as caught and prove nothing.
mutate "control: the page stride is reassociated" miss mla \
    '        let per = u64::from(self.page_size) * elem;
        [
            PageBuffer {
                name: "ckv",
                page_bytes: per * u64::from(self.kv_lora_rank),' \
    '        let per = elem * u64::from(self.page_size);
        [
            PageBuffer {
                name: "ckv",
                page_bytes: u64::from(self.kv_lora_rank) * per,'

echo
echo "mutations: $pass as expected, $fail unexpected"
[[ "$fail" -eq 0 ]]
