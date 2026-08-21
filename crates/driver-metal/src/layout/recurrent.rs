//! The recurrent stack's slab geometry — how big a linear-attention state is,
//! and how many of them fit.
//!
//! A gated-DeltaNet layer carries two states per request, and neither is a KV
//! page:
//!
//! * the **conv window** — the last `Kc` rows of the mixed q|k|v bank, which
//!   the depthwise conv slides over. `Kc * conv_dim` floats.
//! * the **recurrent state** — the DeltaNet memory matrix, `Hv * Vd * Dk`
//!   floats, read-modify-written in place by the one simdgroup that owns
//!   each row.
//!
//! Both are per LAYER and per SLOT, and a slot is a request's seat for its
//! whole life rather than a page it borrows for one step. That is the whole
//! difference from [`kv`](crate::layout::kv): a paged pool hands out pages by
//! the token, this one hands out seats by the request.
//!
//! # Why there are two conv planes and one state plane
//!
//! `gdn_core.metal` cannot shift the conv window in place — `convsilu` reads
//! the taps while the writeback shifts them, from different threadgroups —
//! so the shifted window has to land in a second plane. The kernel takes both
//! (`conv_state` read-only, `new_conv_state` written) and indexes them with
//! the same slot.
//!
//! What it does NOT do is alternate between them: the read plane is always
//! `conv_state`, so after a fire the shifted windows are in the wrong one and
//! somebody has to carry them back. That somebody is the driver, once per
//! layer per fire, over the WHOLE plane.
//!
//! Swapping the two binds each fire is the obvious way to avoid the copy and
//! it is wrong, in a way worth stating because it looks right. A bind is one
//! address for every row of a batch, while which plane holds a slot's live
//! window is per SLOT: a request that sat out a fire did not get its window
//! copied to the other plane, so after one swap it reads a window one step
//! stale, and after the next it reads one two steps stale. Carrying the whole
//! plane back keeps both planes identical for every slot the fire did not
//! touch, which is exactly the invariant a swap breaks.
//!
//! The whole plane rather than the touched slots because a fire's slots are
//! scattered and thirty scattered blits cost more setup than one contiguous
//! copy of a plane this size — 128 KiB a slot on qwen3.6, against a 2 MiB
//! state plane beside it. The recurrent state needs none of this: it is
//! read-modify-written in place by the one simdgroup that owns each row.
//!
//! # Why the allocation is next door
//!
//! Same cut as `layout::kv`: [`Shape`] is arithmetic over integers and is
//! correct with no GPU, so it lives here; `pools::recurrent` turns it into
//! memory and lives behind the one Apple gate.

/// What a driver must allocate to run a recurrent stack.
///
/// Every field is a count, not a byte total — the byte totals are the methods,
/// so there is one place that multiplies and one place that can be wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// How many layers of the stack are linear-attention layers.
    ///
    /// Not the stack's depth: a hybrid interleaves full-attention layers that
    /// carry KV pages and no state at all, and those allocate nothing here.
    pub linear_layers: u32,
    /// Conv channel count — the width of the mixed q|k|v bank.
    pub conv_dim: u32,
    /// Conv kernel width, the window's depth in rows.
    pub conv_k: u32,
    /// Value heads.
    pub v_heads: u32,
    /// Value head dim.
    pub v_dim: u32,
    /// Key head dim — the recurrent state's inner extent.
    pub k_dim: u32,
    /// How many requests can hold a seat at once.
    pub slots: u32,
}

/// Both states are `float` on this backend.
///
/// The element width is the KERNEL's property, not the checkpoint's, and this
/// is the one place that says so. `gdn_core.metal` binds `conv_state` and
/// `rstate` as `device float*` with no template parameter, so a driver that
/// sized these from a model's `state_elem` — which some texts state as 2,
/// meaning the bf16 a CUDA build uses — would allocate half a slab and index
/// off the end of it on the first slot past zero.
pub const ELEM_BYTES: u64 = 4;

/// Two conv planes per layer: the one a fire reads and the one it writes.
///
/// A count rather than a `+ conv` in one expression, because it is the thing
/// a reader doubts: the planes are the same size, the kernel takes both, and
/// the second is not scratch that could be shared between layers — the carry
/// back happens after the whole fire, so every layer's second plane is still
/// live when the next layer runs.
pub const CONV_PLANES: u64 = 2;

impl Shape {
    /// Bytes of one slot's conv window — the stride the kernel indexes a
    /// conv plane with.
    pub fn conv_bytes_per_slot(&self) -> u64 {
        u64::from(self.conv_k) * u64::from(self.conv_dim) * ELEM_BYTES
    }

    /// Bytes one slot's recurrent state occupies in one layer.
    pub fn state_bytes_per_slot(&self) -> u64 {
        u64::from(self.v_heads) * u64::from(self.v_dim) * u64::from(self.k_dim) * ELEM_BYTES
    }

    /// Bytes of ONE of a layer's two conv planes.
    pub fn conv_bytes_per_layer(&self) -> u64 {
        self.conv_bytes_per_slot() * u64::from(self.slots)
    }

    /// Bytes of one layer's whole recurrent-state plane.
    pub fn state_bytes_per_layer(&self) -> u64 {
        self.state_bytes_per_slot() * u64::from(self.slots)
    }

    /// Bytes one slot costs across the WHOLE stack — what the ABI's
    /// `rs_cache_slot_bytes` means, and what a scheduler divides its budget by.
    pub fn bytes_per_slot(&self) -> u64 {
        u64::from(self.linear_layers)
            * (CONV_PLANES * self.conv_bytes_per_slot() + self.state_bytes_per_slot())
    }

    /// Bytes of the entire pool.
    pub fn total_bytes(&self) -> u64 {
        u64::from(self.slots) * self.bytes_per_slot()
    }

    /// The same shape with as many slots as `budget` bytes will hold.
    ///
    /// Returns `None` when one slot does not fit — a caller that got `None`
    /// cannot run the model at all and should say so rather than allocate a
    /// zero-slot pool and fail on the first request.
    pub fn slots_within(&self, budget: u64) -> Option<Self> {
        let per = self.bytes_per_slot();
        if per == 0 {
            return None;
        }
        let slots = u32::try_from(budget / per).unwrap_or(u32::MAX);
        (slots > 0).then_some(Self { slots, ..*self })
    }

    /// The offset of one slot's window within a conv plane.
    pub fn conv_offset(&self, slot: u32) -> u64 {
        u64::from(slot) * self.conv_bytes_per_slot()
    }

    /// The state-plane offset of one slot, in bytes from the plane's base.
    pub fn state_offset(&self, slot: u32) -> u64 {
        u64::from(slot) * self.state_bytes_per_slot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.6-35B-A3B's gated-DeltaNet layers, as the checkpoint states them.
    fn qwen35b() -> Shape {
        Shape {
            linear_layers: 30,
            conv_dim: 8192,
            conv_k: 4,
            v_heads: 32,
            v_dim: 128,
            k_dim: 128,
            slots: 1,
        }
    }

    #[test]
    fn a_conv_window_is_the_kernels_stride() {
        let s = qwen35b();
        assert_eq!(s.conv_bytes_per_slot(), 4 * 8192 * 4);
        assert_eq!(s.conv_offset(7), 7 * s.conv_bytes_per_slot());
    }

    #[test]
    fn a_slots_cost_counts_both_conv_planes() {
        let s = qwen35b();
        let per_layer = 2 * 4 * 8192 * 4 + 32 * 128 * 128 * 4;
        assert_eq!(s.bytes_per_slot(), 30 * per_layer);
        assert_eq!(s.total_bytes(), s.bytes_per_slot());
        // The read plane alone is half the conv cost, which is what an
        // allocation asks for one at a time.
        assert_eq!(s.conv_bytes_per_layer(), 4 * 8192 * 4);
    }

    #[test]
    fn a_budget_buys_whole_slots_and_never_a_fraction_of_one() {
        let s = qwen35b();
        let per = s.bytes_per_slot();
        assert_eq!(s.slots_within(per * 4 + per / 2).unwrap().slots, 4);
        assert_eq!(s.slots_within(per).unwrap().slots, 1);
        assert!(s.slots_within(per - 1).is_none());
    }

    #[test]
    fn state_rows_do_not_overlap() {
        let s = Shape {
            slots: 8,
            ..qwen35b()
        };
        assert_eq!(
            s.state_offset(1) - s.state_offset(0),
            s.state_bytes_per_slot()
        );
        assert_eq!(s.state_offset(8), s.state_bytes_per_layer());
    }

    #[test]
    fn a_stack_with_no_linear_layers_costs_nothing_and_sells_no_slots() {
        let s = Shape {
            linear_layers: 0,
            ..qwen35b()
        };
        assert_eq!(s.bytes_per_slot(), 0);
        assert_eq!(s.total_bytes(), 0);
        assert!(s.slots_within(1 << 30).is_none());
    }
}
