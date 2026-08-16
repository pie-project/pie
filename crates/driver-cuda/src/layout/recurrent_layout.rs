//! Slot addressing for recurrent-state cache.
//! Linear-attention layers are packed densely among full-attention layers.
//! Addresses use `linear_index * layer_stride + slot * slot_stride`.
//! `conv_state` is u16; `recurrent_state` is f32 or bf16 at runtime.

/// The per-slot geometry of one recurrent state cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentStateLayout {
    /// Dense index of layer `i`, or `None` for a full-attention layer.
    linear_layer_index: Vec<Option<u32>>,
    num_linear_layers: u32,
    max_slots: u32,
    conv_dim: u32,
    conv_kernel: u32,
    v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    hidden_size: u32,
    recurrent_is_bf16: bool,
}

/// Shape numbers used to build a [`RecurrentStateLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentShape {
    /// Channels in the causal-conv state.
    pub conv_dim: u32,
    /// Taps in the causal-conv kernel; the conv state holds this many steps.
    pub conv_kernel: u32,
    /// Value heads in the linear-attention state.
    pub v_heads: u32,
    /// Key dimension per head.
    pub head_k_dim: u32,
    /// Value dimension per head.
    pub head_v_dim: u32,
    /// Width of the pending MTP hidden row, or 0 when there is none.
    pub hidden_size: u32,
    /// Concurrent slots. Clamped up to 1 by [`RecurrentStateLayout::new`].
    pub max_slots: u32,
    /// Whether the recurrent state is stored as `bf16` rather than `f32`.
    pub recurrent_is_bf16: bool,
}

/// Where one layer/slot's state lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotAddr {
    /// Byte offset from the base of the corresponding pooled allocation.
    pub offset: u64,
    /// Bytes belonging to this slot.
    pub len: u64,
}

impl RecurrentStateLayout {
    /// Build a layout; `max_slots` is clamped up to 1.
    #[must_use]
    pub fn new(layer_is_linear: &[bool], shape: RecurrentShape) -> Self {
        let mut next = 0u32;
        let linear_layer_index = layer_is_linear
            .iter()
            .map(|&is_linear| {
                if is_linear {
                    let idx = next;
                    next += 1;
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        Self {
            linear_layer_index,
            num_linear_layers: next,
            max_slots: shape.max_slots.max(1),
            conv_dim: shape.conv_dim,
            conv_kernel: shape.conv_kernel,
            v_heads: shape.v_heads,
            head_k_dim: shape.head_k_dim,
            head_v_dim: shape.head_v_dim,
            hidden_size: shape.hidden_size,
            recurrent_is_bf16: shape.recurrent_is_bf16,
        }
    }

    /// Force `bf16` recurrent storage; layer compaction is dtype-independent.
    pub const fn force_recurrent_bf16(&mut self) {
        self.recurrent_is_bf16 = true;
    }

    /// Channels in the causal-conv state.
    #[must_use]
    pub const fn conv_dim(&self) -> u32 {
        self.conv_dim
    }

    /// Taps in the causal-conv kernel.
    #[must_use]
    pub const fn conv_kernel(&self) -> u32 {
        self.conv_kernel
    }

    /// Value heads in the linear-attention state.
    #[must_use]
    pub const fn v_heads(&self) -> u32 {
        self.v_heads
    }

    /// Key dimension per head.
    #[must_use]
    pub const fn head_k_dim(&self) -> u32 {
        self.head_k_dim
    }

    /// Value dimension per head.
    #[must_use]
    pub const fn head_v_dim(&self) -> u32 {
        self.head_v_dim
    }

    /// Width of the MTP pending-hidden row, or 0 when there is none.
    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// Total layers in the stack, linear and full-attention alike.
    #[must_use]
    pub fn num_layers(&self) -> u32 {
        self.linear_layer_index.len() as u32
    }
    /// How many layers actually carry recurrent state.
    #[must_use]
    pub const fn num_linear_layers(&self) -> u32 {
        self.num_linear_layers
    }
    /// Concurrent request slots, at least 1.
    #[must_use]
    pub const fn max_slots(&self) -> u32 {
        self.max_slots
    }
    /// Is the recurrent state stored as bf16 rather than f32?
    #[must_use]
    pub const fn recurrent_is_bf16(&self) -> bool {
        self.recurrent_is_bf16
    }

    /// Is this layer a linear-attention layer?
    #[must_use]
    pub fn is_linear(&self, layer: u32) -> bool {
        self.linear_index(layer).is_some()
    }

    /// The dense index of a layer among the linear ones.
    #[must_use]
    pub fn linear_index(&self, layer: u32) -> Option<u32> {
        self.linear_layer_index
            .get(layer as usize)
            .copied()
            .flatten()
    }

    /// Bytes between consecutive slots of `conv_state`. Always `u16`.
    #[must_use]
    pub const fn conv_slot_stride_bytes(&self) -> u64 {
        self.conv_kernel as u64 * self.conv_dim as u64 * 2
    }

    /// Elements between consecutive slots of `recurrent_state`.
    #[must_use]
    pub const fn recurrent_slot_stride_elems(&self) -> u64 {
        self.v_heads as u64 * self.head_k_dim as u64 * self.head_v_dim as u64
    }

    /// Bytes between `recurrent_state` slots, using f32 or bf16 width.
    #[must_use]
    pub const fn recurrent_slot_stride_bytes(&self) -> u64 {
        self.recurrent_slot_stride_elems() * if self.recurrent_is_bf16 { 2 } else { 4 }
    }

    /// Address of one layer/slot's convolution state; `None` for full-attention, panics
    /// if out of range.
    #[must_use]
    pub fn conv_state(&self, layer: u32, slot: u32) -> Option<SlotAddr> {
        let idx = self.checked_index(layer, slot)?;
        let stride = self.conv_slot_stride_bytes();
        Some(SlotAddr {
            offset: u64::from(idx) * u64::from(self.max_slots) * stride + u64::from(slot) * stride,
            len: stride,
        })
    }

    /// Address of one layer/slot's recurrent state; `None` for full-attention, panics
    /// if out of range.
    #[must_use]
    pub fn recurrent_state(&self, layer: u32, slot: u32) -> Option<SlotAddr> {
        let idx = self.checked_index(layer, slot)?;
        let stride = self.recurrent_slot_stride_bytes();
        Some(SlotAddr {
            offset: u64::from(idx) * u64::from(self.max_slots) * stride + u64::from(slot) * stride,
            len: stride,
        })
    }

    /// Address of one pending-MTP hidden row; `None` when unconfigured, panics if out of range.
    #[must_use]
    pub fn mtp_pending_hidden(&self, slot: u32) -> Option<SlotAddr> {
        assert!(
            slot < self.max_slots,
            "slot {slot} of {} slots",
            self.max_slots
        );
        if self.hidden_size == 0 {
            return None;
        }
        let stride = u64::from(self.hidden_size) * 2;
        Some(SlotAddr {
            offset: u64::from(slot) * stride,
            len: stride,
        })
    }

    /// Total bytes of the pooled `conv_state` allocation.
    #[must_use]
    pub const fn conv_total_bytes(&self) -> u64 {
        self.conv_slot_stride_bytes() * self.max_slots as u64 * self.num_linear_layers as u64
    }

    /// Total bytes of the pooled `recurrent_state` allocation.
    #[must_use]
    pub const fn recurrent_total_bytes(&self) -> u64 {
        self.recurrent_slot_stride_bytes() * self.max_slots as u64 * self.num_linear_layers as u64
    }

    /// Total bytes of the MTP hidden pool.
    #[must_use]
    pub const fn mtp_total_bytes(&self) -> u64 {
        self.hidden_size as u64 * 2 * self.max_slots as u64
    }

    /// Every byte the cache allocates.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.conv_total_bytes() + self.recurrent_total_bytes() + self.mtp_total_bytes()
    }

    /// Bytes one request occupies across all linear layers.
    #[must_use]
    pub const fn bytes_per_slot(&self) -> u64 {
        (self.conv_slot_stride_bytes() + self.recurrent_slot_stride_bytes())
            * self.num_linear_layers as u64
            + self.hidden_size as u64 * 2
    }

    /// Bounds-check, then resolve; `None` only means a full-attention layer.
    fn checked_index(&self, layer: u32, slot: u32) -> Option<u32> {
        assert!(
            slot < self.max_slots,
            "slot {slot} of {} slots",
            self.max_slots
        );
        assert!(
            (layer as usize) < self.linear_layer_index.len(),
            "layer {layer} of {} layers",
            self.linear_layer_index.len()
        );
        self.linear_index(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hybrid shape where dense index and layer number diverge.
    fn hybrid() -> RecurrentStateLayout {
        let linear = [true, false, false, true, false, false, false, true];
        RecurrentStateLayout::new(
            &linear,
            RecurrentShape {
                conv_dim: 4096,
                conv_kernel: 4,
                v_heads: 32,
                head_k_dim: 128,
                head_v_dim: 128,
                hidden_size: 2048,
                max_slots: 16,
                recurrent_is_bf16: false,
            },
        )
    }

    #[test]
    fn linear_layers_are_packed_densely_not_addressed_by_layer_number() {
        let l = hybrid();
        assert_eq!(l.num_linear_layers(), 3);
        assert_eq!(l.linear_index(0), Some(0));
        assert_eq!(l.linear_index(3), Some(1));
        assert_eq!(l.linear_index(7), Some(2));
        for full in [1, 2, 4, 5, 6] {
            assert_eq!(l.linear_index(full), None, "layer {full}");
        }
    }

    #[test]
    fn the_last_linear_layer_addresses_by_its_dense_index() {
        // Pins dense linear-index addressing, not raw layer-number addressing.
        let l = hybrid();
        let stride = l.conv_slot_stride_bytes();
        let layer_stride = stride * u64::from(l.max_slots());
        assert_eq!(l.conv_state(7, 0).unwrap().offset, 2 * layer_stride);
        assert!(l.conv_state(7, 15).unwrap().offset < l.conv_total_bytes());
    }

    #[test]
    fn every_addressable_slot_lies_inside_its_allocation() {
        let l = hybrid();
        for layer in 0..l.num_layers() {
            for slot in 0..l.max_slots() {
                if let Some(a) = l.conv_state(layer, slot) {
                    assert!(
                        a.offset + a.len <= l.conv_total_bytes(),
                        "conv {layer}/{slot}"
                    );
                }
                if let Some(a) = l.recurrent_state(layer, slot) {
                    assert!(
                        a.offset + a.len <= l.recurrent_total_bytes(),
                        "rec {layer}/{slot}"
                    );
                }
            }
        }
    }

    #[test]
    fn distinct_layer_slot_pairs_never_alias() {
        let l = hybrid();
        let mut seen = Vec::new();
        for layer in 0..l.num_layers() {
            for slot in 0..l.max_slots() {
                if let Some(a) = l.conv_state(layer, slot) {
                    seen.push(a.offset);
                }
            }
        }
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "two slots resolved to the same offset");
        assert_eq!(total, 3 * 16);
    }

    #[test]
    fn the_recurrent_stride_follows_the_dtype_switch() {
        let linear = [true, true];
        let f32_layout = RecurrentStateLayout::new(
            &linear,
            RecurrentShape {
                conv_dim: 4096,
                conv_kernel: 4,
                v_heads: 32,
                head_k_dim: 128,
                head_v_dim: 128,
                hidden_size: 0,
                max_slots: 4,
                recurrent_is_bf16: false,
            },
        );
        let bf16_layout = RecurrentStateLayout::new(
            &linear,
            RecurrentShape {
                conv_dim: 4096,
                conv_kernel: 4,
                v_heads: 32,
                head_k_dim: 128,
                head_v_dim: 128,
                hidden_size: 0,
                max_slots: 4,
                recurrent_is_bf16: true,
            },
        );
        assert_eq!(f32_layout.recurrent_slot_stride_bytes(), 32 * 128 * 128 * 4);
        assert_eq!(
            bf16_layout.recurrent_slot_stride_bytes(),
            32 * 128 * 128 * 2
        );
        assert_eq!(
            f32_layout.recurrent_slot_stride_elems(),
            bf16_layout.recurrent_slot_stride_elems(),
            "the element count is the same either way; only the width moves"
        );
        // Conv state is always u16, independent of the recurrent dtype.
        assert_eq!(
            f32_layout.conv_slot_stride_bytes(),
            bf16_layout.conv_slot_stride_bytes()
        );
    }

    #[test]
    fn a_full_attention_layer_has_no_state_which_is_not_an_error() {
        let l = hybrid();
        assert_eq!(l.conv_state(1, 0), None);
        assert_eq!(l.recurrent_state(1, 0), None);
    }

    #[test]
    #[should_panic(expected = "slot 16 of 16 slots")]
    fn an_out_of_range_slot_is_a_caller_bug() {
        let _ = hybrid().conv_state(0, 16);
    }

    #[test]
    #[should_panic(expected = "layer 99 of 8 layers")]
    fn an_out_of_range_layer_is_a_caller_bug() {
        let _ = hybrid().recurrent_state(99, 0);
    }

    #[test]
    #[should_panic(expected = "slot 16 of 16 slots")]
    fn an_out_of_range_mtp_slot_is_a_caller_bug() {
        let _ = hybrid().mtp_pending_hidden(16);
    }

    #[test]
    fn max_slots_of_zero_is_treated_as_one() {
        let l = RecurrentStateLayout::new(
            &[true],
            RecurrentShape {
                conv_dim: 8,
                conv_kernel: 4,
                v_heads: 1,
                head_k_dim: 8,
                head_v_dim: 8,
                hidden_size: 16,
                max_slots: 0,
                recurrent_is_bf16: false,
            },
        );
        assert_eq!(l.max_slots(), 1);
        assert!(l.conv_state(0, 0).is_some());
    }

    #[test]
    fn a_model_with_no_hidden_state_has_no_mtp_pool() {
        let l = RecurrentStateLayout::new(
            &[true],
            RecurrentShape {
                conv_dim: 8,
                conv_kernel: 4,
                v_heads: 1,
                head_k_dim: 8,
                head_v_dim: 8,
                hidden_size: 0,
                max_slots: 4,
                recurrent_is_bf16: false,
            },
        );
        assert_eq!(l.mtp_pending_hidden(0), None);
        assert_eq!(l.mtp_total_bytes(), 0);
    }

    #[test]
    fn a_stack_with_no_linear_layers_allocates_nothing_for_state() {
        let l = RecurrentStateLayout::new(
            &[false; 8],
            RecurrentShape {
                conv_dim: 4096,
                conv_kernel: 4,
                v_heads: 32,
                head_k_dim: 128,
                head_v_dim: 128,
                hidden_size: 0,
                max_slots: 16,
                recurrent_is_bf16: false,
            },
        );
        assert_eq!(l.conv_state(0, 0), None);
        assert_eq!(l.num_linear_layers(), 0);
        assert_eq!(l.conv_total_bytes(), 0);
        assert_eq!(l.recurrent_total_bytes(), 0);
        assert_eq!(l.total_bytes(), 0);
        assert!(!l.is_linear(0));
    }

    #[test]
    fn totals_are_the_per_slot_cost_times_the_slot_count() {
        let l = hybrid();
        assert_eq!(
            l.total_bytes(),
            l.bytes_per_slot() * u64::from(l.max_slots())
        );
    }
}
