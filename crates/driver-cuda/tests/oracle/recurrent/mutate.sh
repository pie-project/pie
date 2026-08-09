#!/bin/bash
# Mutation-tests the golden in tests/recurrent_parity.rs.
#
# A golden proves the two implementations agree; it does not prove the sweep
# would NOTICE a disagreement. Each mutation below is a plausible porting slip
# applied to the Rust. The golden must reject every one of them.
#
# The final entry is a NO-OP control that must still COMPILE and must MISS. A
# mutation that fails to build registers as caught by any harness at all, so a
# non-compiling control proves nothing about this one.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(cd "$HERE/../../.." && pwd)"
CACHE="$CRATE/src/store/recurrent_state_cache.rs"
LAYOUT="$CRATE/src/store/recurrent_layout.rs"
B1="$(mktemp)"; B2="$(mktemp)"
cp "$CACHE" "$B1"; cp "$LAYOUT" "$B2"
restore() { cp "$B1" "$CACHE"; cp "$B2" "$LAYOUT"; rm -f "$B1" "$B2"; }
trap restore EXIT

pass=0
fail=0

# $1 label, $2 expectation (catch|miss), $3 file (cache|layout), $4 from, $5 to
mutate() {
    local label="$1" expect="$2" file="$3" from="$4" to="$5"
    cp "$B1" "$CACHE"; cp "$B2" "$LAYOUT"
    local target="$CACHE"
    [[ "$file" == "layout" ]] && target="$LAYOUT"
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
    if (cd "$CRATE" && cargo test --features cuda-13 --test recurrent_parity \
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

echo "mutating $CACHE and $LAYOUT ..."

# --- the dense linear-layer compaction --------------------------------------
mutate "compaction: full-attention layers get an index too" catch layout \
    '                if is_linear {
                    let idx = next;
                    next += 1;
                    Some(idx)
                } else {
                    None
                }' \
    '                {
                    let idx = next;
                    next += 1;
                    is_linear.then_some(idx)
                }'
mutate "accessor indexes by transformer layer, not linear index" catch cache \
    '            .map(|idx| (idx, slot.unsigned_abs())))' \
    '            .map(|_| (layer.unsigned_abs(), slot.unsigned_abs())))'

# --- the strides ------------------------------------------------------------
mutate "conv stride charges 4 bytes per element" catch layout \
    'self.conv_kernel as u64 * self.conv_dim as u64 * 2' \
    'self.conv_kernel as u64 * self.conv_dim as u64 * 4'
mutate "recurrent stride ignores the bf16 flag" catch layout \
    'if self.recurrent_is_bf16 { 2 } else { 4 }' \
    '4'
mutate "recurrent stride drops head_v_dim" catch layout \
    'self.v_heads as u64 * self.head_k_dim as u64 * self.head_v_dim as u64' \
    'self.v_heads as u64 * self.head_k_dim as u64'
mutate "conv_dim and conv_kernel stored transposed" catch layout \
    '            conv_dim: shape.conv_dim,
            conv_kernel: shape.conv_kernel,' \
    '            conv_dim: shape.conv_kernel,
            conv_kernel: shape.conv_dim,'

# --- the pitched slot operations --------------------------------------------
mutate "reset_slot pitch is the slot stride, not the layer stride" catch cache \
    '                    pitch: stride * slots,
                    width: stride,
                    rows: layers,
                });
            }
        }
        if self.has_mtp_hidden() {
            let hidden' \
    '                    pitch: stride,
                    width: stride,
                    rows: layers,
                });
            }
        }
        if self.has_mtp_hidden() {
            let hidden'
mutate "reset_slot offsets by the layer stride" catch cache \
    '                    offset: u64::from(slot.unsigned_abs()) * stride,' \
    '                    offset: u64::from(slot.unsigned_abs()) * stride * slots,'
mutate "reset_slot zeroes the whole run, not one slot" catch cache \
    '                    width: stride,
                    rows: layers,
                });
            }
        }
        if self.has_mtp_hidden() {
            let hidden' \
    '                    width: stride * slots,
                    rows: layers,
                });
            }
        }
        if self.has_mtp_hidden() {
            let hidden'
mutate "reset_slot rows counts all layers, not the linear ones" catch cache \
    '        let layers = u64::from(self.layout.num_linear_layers());
        if layers > 0 {
            for (buffer, stride) in self.state_strides() {
                ops.push(StateOp::Memset2D {' \
    '        let layers = u64::from(self.layout.num_layers());
        if layers > 0 {
            for (buffer, stride) in self.state_strides() {
                ops.push(StateOp::Memset2D {'

# --- the copies -------------------------------------------------------------
mutate "copy src and dst transposed" catch cache \
    '                dst: u64::from(dst_slot.unsigned_abs()) * stride,
                src: u64::from(src_slot.unsigned_abs()) * stride,' \
    '                dst: u64::from(src_slot.unsigned_abs()) * stride,
                src: u64::from(dst_slot.unsigned_abs()) * stride,'
mutate "the linear-only copy also moves the MTP row" catch cache \
    '        let mut ops = self.slot_copy_ops(src_slot, dst_slot);
        if src_slot != dst_slot && self.has_mtp_hidden() {' \
    '        let mut ops = Vec::new();
        if src_slot != dst_slot && self.has_mtp_hidden() {'
mutate "self-copy still issues the transfers" catch cache \
    'if src_slot == dst_slot || layers == 0 {
            return Vec::new();
        }' \
    'if layers == 0 {
            return Vec::new();
        }'
mutate "the two copies report the same exception name" catch cache \
    '            "RecurrentStateCache::copy_linear_state_slot_d2d",
        )?;
        Ok(self.slot_copy_ops(src_slot, dst_slot))' \
    '            "RecurrentStateCache::copy_slot_d2d",
        )?;
        Ok(self.slot_copy_ops(src_slot, dst_slot))'

# --- the device-predicated reset --------------------------------------------
mutate "the MTP tier gets one row per layer" catch cache \
    '                slot_bytes: hidden_bytes,
                row_pitch: hidden_bytes * slots,
                rows: 1,' \
    '                slot_bytes: hidden_bytes,
                row_pitch: hidden_bytes * slots,
                rows: layers.max(1),'
mutate "a zero request count still launches the kernel" catch cache \
    'if slot_ids.is_none() || is_fresh.is_none() || request_count <= 0 {' \
    'if false {'
mutate "a null slot_ids array still launches the kernel" catch cache \
    'if slot_ids.is_none() || is_fresh.is_none() || request_count <= 0 {' \
    'if is_fresh.is_none() || request_count <= 0 {'
mutate "a null is_fresh array still launches the kernel" catch cache \
    'if slot_ids.is_none() || is_fresh.is_none() || request_count <= 0 {' \
    'if slot_ids.is_none() || request_count <= 0 {'
mutate "a negative request count is taken as positive" catch cache \
    'if slot_ids.is_none() || is_fresh.is_none() || request_count <= 0 {' \
    'if slot_ids.is_none() || is_fresh.is_none() || request_count == 0 {'

# --- the clamps -------------------------------------------------------------
mutate "max_slots is not clamped up to 1" catch layout \
    'max_slots: shape.max_slots.max(1),' \
    'max_slots: shape.max_slots,'
mutate "hidden_size is not clamped up to 0" catch cache \
    'let hidden_size = hidden_size.max(0).unsigned_abs();' \
    'let hidden_size = hidden_size.unsigned_abs();'
mutate "the bf16 default is false" catch cache \
    'pub const fn recurrent_state_bf16_default() -> bool {
    true
}' \
    'pub const fn recurrent_state_bf16_default() -> bool {
    false
}'
# Expected to MISS, and that is a finding rather than a hole in the sweep.
# `recurrent_state_bf16_default()` is now a constant `true`, so the flag is
# already set and forcing it changes nothing -- which also means the C++'s own
# re-allocation branch in `allocate_bf16_recurrent` is dead, and the whole
# constructor is currently indistinguishable from `allocate(hidden_size=0)`.
mutate "forcing bf16 is redundant while the default is already bf16" miss cache \
    'c.layout.force_recurrent_bf16();' \
    'let _ = &mut c;'
mutate "the MTP tier exists at hidden_size 0" catch cache \
    'self.layout.hidden_size() > 0' \
    'true'

# --- validation order and messages ------------------------------------------
mutate "layer is checked before slot" catch cache \
    '        if slot < 0 || slot.unsigned_abs() >= self.layout.max_slots() {
            return Err(Error::invalid(who, "slot out of range"));
        }
        if layer < 0 || layer.unsigned_abs() >= self.layout.num_layers() {
            return Err(Error::invalid(who, "layer out of range"));
        }' \
    '        if layer < 0 || layer.unsigned_abs() >= self.layout.num_layers() {
            return Err(Error::invalid(who, "layer out of range"));
        }
        if slot < 0 || slot.unsigned_abs() >= self.layout.max_slots() {
            return Err(Error::invalid(who, "slot out of range"));
        }'
mutate "recurrent_state_raw reports its own name" catch cache \
    'self.checked_index(layer, slot, "RecurrentStateCache::recurrent_state")' \
    'self.checked_index(layer, slot, "RecurrentStateCache::recurrent_state_raw")'
mutate "the fp32 accessor checks its arguments first" catch cache \
    '        if self.layout.recurrent_is_bf16() {
            return Err(Error::invalid(
                "RecurrentStateCache::recurrent_state",
                "recurrent state is bf16",
            ));
        }
        self.recurrent_state_raw(layer, slot)' \
    '        let raw = self.recurrent_state_raw(layer, slot)?;
        if self.layout.recurrent_is_bf16() {
            return Err(Error::invalid(
                "RecurrentStateCache::recurrent_state",
                "recurrent state is bf16",
            ));
        }
        Ok(raw)'
mutate "the last slot is rejected (off-by-one)" catch cache \
    'if slot < 0 || slot.unsigned_abs() >= self.layout.max_slots() {
            return Err(Error::invalid(who, "slot out of range"));
        }
        Ok(())' \
    'if slot < 0 || slot.unsigned_abs() + 1 >= self.layout.max_slots() {
            return Err(Error::invalid(who, "slot out of range"));
        }
        Ok(())'

# --- the optional tiers -----------------------------------------------------
mutate "the stash cap can also raise the token count" catch cache \
    '            && want.unsigned_abs() < max_tokens' \
    '            && want.unsigned_abs() != max_tokens'
mutate "the stash cap accepts zero" catch cache \
    '        if let Some(want) = stash_tokens_cap
            && want > 0' \
    '        if let Some(want) = stash_tokens_cap
            && want >= 0'
mutate "the cap parser rejects trailing garbage instead of stopping" catch cache \
    '        let Some(d) = c.to_digit(10) else { break };' \
    '        let Some(d) = c.to_digit(10) else { return Some(0) };'
mutate "the cap parser ignores a leading plus sign" catch cache \
    '        None => (1i64, t.strip_prefix(0x2b as char).unwrap_or(t)),' \
    '        None => (1i64, t),'
mutate "the cap parser does not skip leading whitespace" catch cache \
    'let t = raw.trim_start();' \
    'let t = raw.as_str();'
mutate "a stack with no linear layers still gets a stash" catch cache \
    'if max_tokens == 0 || hidden == 0 || self.layout.num_linear_layers() == 0 {
            return;
        }' \
    'if max_tokens == 0 || hidden == 0 {
            return;
        }'
mutate "the pool accepts a zero slot count" catch cache \
    'if page_tokens == 0 || hidden == 0 || num_slots == 0 || self.layout.num_linear_layers() == 0' \
    'if page_tokens == 0 || hidden == 0 || self.layout.num_linear_layers() == 0'
mutate "the pool layer stride uses max_slots instead of its own" catch cache \
    'Some(u64::from(linear_idx) * per_slot * u64::from(dims.num_slots) + u64::from(slot) * per_slot)' \
    'Some(u64::from(linear_idx) * per_slot * u64::from(self.layout.max_slots()) + u64::from(slot) * per_slot)'
mutate "the pool slot bound is its layer bound" catch cache \
    'if linear_idx >= self.layout.num_linear_layers() || slot >= dims.num_slots {' \
    'if linear_idx >= self.layout.num_linear_layers() || slot >= self.layout.max_slots() {'
mutate "the stash is indexed by transformer layer count" catch cache \
    '        if linear_idx >= self.layout.num_linear_layers() {
            return None;
        }
        Some(u64::from(linear_idx) * u64::from(dims.max_tokens)' \
    '        if linear_idx >= self.layout.num_layers() {
            return None;
        }
        Some(u64::from(linear_idx) * u64::from(dims.max_tokens)'

# --- the buffer order -------------------------------------------------------
mutate "conv and recurrent are issued in the other order" catch cache \
    '            (Buffer::Conv, self.layout.conv_slot_stride_bytes()),
            (Buffer::Recurrent, self.layout.recurrent_slot_stride_bytes()),' \
    '            (Buffer::Recurrent, self.layout.recurrent_slot_stride_bytes()),
            (Buffer::Conv, self.layout.conv_slot_stride_bytes()),'

# --- the control ------------------------------------------------------------
mutate "CONTROL: rewrite a product with an explicit 1 factor (no-op)" miss cache \
    'len: u64::from(self.layout.hidden_size()) * slots * U16,' \
    'len: 1 * u64::from(self.layout.hidden_size()) * slots * U16,'

echo
echo "mutations: $pass as expected, $fail unexpected"
[[ "$fail" -eq 0 ]]
