//! The per-fire forward workspace: reusable scratch every llama-like forward
//! path writes into, sized once against `max_tokens`.
//!
//! [`WorkspaceLayout::slots`] is the single statement of the layout; `bytes`
//! sums it and `specs` hands it to the allocator, so the planner's subtracted
//! figure and the tensors allocated cannot drift apart.

use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::TensorSpec;

/// A named buffer in the workspace. Declaration order is allocation order,
/// which the parity transcript records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Slot {
    /// `[N, hidden]` — the residual stream.
    Y,
    /// `[N, hidden]` — pre-attention norm output.
    NormX,
    /// `[N, hidden]` — saved verifier hidden rows for speculative decode.
    SpecHidden,
    /// `[N, Hq + 2*Hk]` — fused QKV projection output.
    QkvFused,
    /// `[N, 2*I]` — fused gate+up projection output.
    GateUpFused,
    /// `[N, 2*hidden]` — Qwen3.6 MTP fc input.
    MtpConcat,
    /// `[1, vocab]` — preserves target row 0 while MTP drafts run.
    MtpRow0Save,
    /// `[N, head_dim]` fp32 — RoPE cos in the first half of each row, sin in
    /// the second.
    RopeTable,
    /// `[N, Hq]` — packed queries.
    Q,
    /// `[N, Hk]` — packed keys.
    K,
    /// `[N, Hk]` — packed values.
    V,
    /// `[N, Hq]` — attention output.
    AttnOut,
    /// `[N, hidden]` — post-attention norm output.
    NormY,
    /// `[N, hidden + intermediate]` — the declared executor's SSA value arena.
    DeclaredValues,
    /// `[N, I]` — MLP gate.
    Gate,
    /// `[N, I]` — MLP up.
    Up,
    /// `[logits_rows, vocab]` — see [`WorkspaceLayout::logits_rows`].
    Logits,
    /// `[output_rows, vocab]` fp32 — softmax scratch for sampling.
    Probs,
    /// `[logits_rows, 1]` i32 — fused-argmax token output.
    SampledTokens,
    /// `[logits_rows, ARGMAX_ACCUM_SLOTS]` fp32 — fused-argmax running values.
    ArgmaxAccVal,
    /// `[logits_rows, ARGMAX_ACCUM_SLOTS]` i32 — fused-argmax running indices.
    ArgmaxAccIdx,
    /// `[N, Hq_pad]` — allocated only when `head_dim != head_dim_kernel`.
    QPadded,
    /// `[N, Hk_pad]` — allocated only when `head_dim != head_dim_kernel`.
    KPadded,
    /// `[N, Hk_pad]` — allocated only when `head_dim != head_dim_kernel`.
    VPadded,
    /// `[N, Hq_pad]` — allocated only when `head_dim != head_dim_kernel`.
    AttnOutPadded,
}

/// Running (value, index) slots per row for the fused-argmax epilogue.
///
/// Must match `kernels::sample::kArgmaxAccumSlots`; the parity oracle checks it.
pub const ARGMAX_ACCUM_SLOTS: i64 = 32;

/// The shape parameters `allocate_full` takes, named so adjacent same-typed
/// args can't be swapped silently at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceShape {
    /// `cfg.hidden_size`.
    pub hidden_size: i64,
    /// `cfg.vocab_size`.
    pub vocab_size: i64,
    /// `cfg.head_dim`.
    pub head_dim: i64,
    /// `cfg.head_dim_kernel` — the extent the attention kernel wants, which
    /// exceeds `head_dim` on Phi-3 (96 rounded to 128).
    pub head_dim_kernel: i64,
    /// `max_tokens`, the row count every per-token buffer is sized by.
    pub max_tokens: i64,
    /// `max_intermediate` — the widest MLP any layer in the stack asks for,
    /// which is not `cfg.intermediate_size` on a mixed stack.
    pub max_intermediate: i64,
    /// `max_Hq` — widest packed query width, `q_heads * head_dim`.
    pub max_hq: i64,
    /// `max_Hk` — widest packed key/value width, `kv_heads * head_dim`.
    pub max_hk: i64,
    /// `max_output_rows`; `0` means "same as `max_tokens`".
    pub max_output_rows: i64,
    /// MTP draft rows reserved at the tail of `logits`. Negative is clamped to zero.
    pub max_mtp_draft_rows: i64,
}

/// The workspace layout for one shape: which buffers exist, how wide, in what
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceLayout {
    shape: WorkspaceShape,
}

impl WorkspaceLayout {
    /// Derive the layout for a shape.
    #[must_use]
    pub const fn new(shape: WorkspaceShape) -> Self {
        Self { shape }
    }

    /// The shape this layout was derived from.
    #[must_use]
    pub const fn shape(&self) -> &WorkspaceShape {
        &self.shape
    }

    /// First `logits` row reserved for MTP drafts: where the target rows end.
    #[must_use]
    pub const fn mtp_draft_row_base(&self) -> i64 {
        self.shape.max_tokens
    }

    /// Rows in `logits`: the target rows plus any MTP draft reserve.
    #[must_use]
    pub const fn logits_rows(&self) -> i64 {
        self.shape.max_tokens + self.draft_rows()
    }

    /// `max_mtp_draft_rows` clamped at zero.
    #[must_use]
    const fn draft_rows(&self) -> i64 {
        if self.shape.max_mtp_draft_rows > 0 {
            self.shape.max_mtp_draft_rows
        } else {
            0
        }
    }

    /// Rows in `probs`: `max_output_rows`, or `max_tokens` when unset.
    #[must_use]
    pub const fn output_rows(&self) -> i64 {
        if self.shape.max_output_rows > 0 {
            self.shape.max_output_rows
        } else {
            self.shape.max_tokens
        }
    }

    /// Whether the packed q/k/v buffers need padded companions.
    #[must_use]
    pub const fn needs_padded_qkv(&self) -> bool {
        self.shape.head_dim != self.shape.head_dim_kernel
    }

    /// Padded query width, or `None` when the packed buffers are used
    /// directly.
    #[must_use]
    pub const fn padded_hq(&self) -> Option<i64> {
        if self.needs_padded_qkv() {
            Some(self.q_heads() * self.shape.head_dim_kernel)
        } else {
            None
        }
    }

    /// Padded key/value width, or `None`.
    #[must_use]
    pub const fn padded_hk(&self) -> Option<i64> {
        if self.needs_padded_qkv() {
            Some(self.kv_heads() * self.shape.head_dim_kernel)
        } else {
            None
        }
    }

    /// Head-count divisor: `head_dim` floored at 1, so a zero head dim
    /// yields zero heads instead of trapping.
    #[must_use]
    const fn head_dim_divisor(&self) -> i64 {
        if self.shape.head_dim >= 1 {
            self.shape.head_dim
        } else {
            1
        }
    }

    /// Query heads, recovered from the packed width.
    #[must_use]
    const fn q_heads(&self) -> i64 {
        self.shape.max_hq / self.head_dim_divisor()
    }

    /// Key/value heads, recovered the same way.
    #[must_use]
    const fn kv_heads(&self) -> i64 {
        self.shape.max_hk / self.head_dim_divisor()
    }

    /// Every buffer in the layout, in allocation order — the single
    /// statement both the allocator and byte budget read.
    #[must_use]
    pub fn slots(&self) -> Vec<(Slot, DType, [i64; 2])> {
        let s = &self.shape;
        let n = s.max_tokens;
        let logits_rows = self.logits_rows();
        let mut out = vec![
            (Slot::Y, DType::Bf16, [n, s.hidden_size]),
            (Slot::NormX, DType::Bf16, [n, s.hidden_size]),
            (Slot::SpecHidden, DType::Bf16, [n, s.hidden_size]),
            (Slot::QkvFused, DType::Bf16, [n, s.max_hq + 2 * s.max_hk]),
            (Slot::GateUpFused, DType::Bf16, [n, 2 * s.max_intermediate]),
            (Slot::MtpConcat, DType::Bf16, [n, 2 * s.hidden_size]),
            (Slot::MtpRow0Save, DType::Bf16, [1, s.vocab_size]),
            (Slot::RopeTable, DType::Fp32, [n, s.head_dim]),
            (Slot::Q, DType::Bf16, [n, s.max_hq]),
            (Slot::K, DType::Bf16, [n, s.max_hk]),
            (Slot::V, DType::Bf16, [n, s.max_hk]),
            (Slot::AttnOut, DType::Bf16, [n, s.max_hq]),
            (Slot::NormY, DType::Bf16, [n, s.hidden_size]),
            (
                Slot::DeclaredValues,
                DType::Bf16,
                [n, s.hidden_size + s.max_intermediate],
            ),
            (Slot::Gate, DType::Bf16, [n, s.max_intermediate]),
            (Slot::Up, DType::Bf16, [n, s.max_intermediate]),
            (Slot::Logits, DType::Bf16, [logits_rows, s.vocab_size]),
            (Slot::Probs, DType::Fp32, [self.output_rows(), s.vocab_size]),
            (Slot::SampledTokens, DType::Int32, [logits_rows, 1]),
            (
                Slot::ArgmaxAccVal,
                DType::Fp32,
                [logits_rows, ARGMAX_ACCUM_SLOTS],
            ),
            (
                Slot::ArgmaxAccIdx,
                DType::Int32,
                [logits_rows, ARGMAX_ACCUM_SLOTS],
            ),
        ];
        if let (Some(hq_pad), Some(hk_pad)) = (self.padded_hq(), self.padded_hk()) {
            out.push((Slot::QPadded, DType::Bf16, [n, hq_pad]));
            out.push((Slot::KPadded, DType::Bf16, [n, hk_pad]));
            out.push((Slot::VPadded, DType::Bf16, [n, hk_pad]));
            out.push((Slot::AttnOutPadded, DType::Bf16, [n, hq_pad]));
        }
        out
    }

    /// The layout as allocatable specs, in the same order.
    pub fn specs(&self) -> Result<Vec<(Slot, TensorSpec)>> {
        self.slots()
            .into_iter()
            .map(|(slot, dtype, shape)| Ok((slot, TensorSpec::new(dtype, shape.to_vec())?)))
            .collect()
    }

    /// What the workspace actually costs.
    ///
    /// Summed from [`Self::slots`], so it is the same layout the allocator
    /// walks by construction — and that "by construction" is why two other
    /// methods are gone.
    ///
    /// # The check that could only pass
    ///
    /// `cpp_budget_bytes` stood beside this claiming to be "derived
    /// independently the way the C++ does so the parity oracle compares two
    /// walks, not a value against itself", and `budget_shortfall` was
    /// `bytes() - cpp_budget_bytes()` — with `workspace_parity` asserting the
    /// difference is zero on every shape in the grid.
    ///
    /// The two bodies were BYTE-IDENTICAL. Both walked `slots()` and summed
    /// `slot_bytes`, so the subtraction was `x - x` and the assertion could
    /// not fail for any input, including the inputs it was written to catch.
    /// The parity test's own doc records when that happened: "They used to be
    /// two hand-written lists and differed by `declared_values +
    /// mtp_row0_save`; both now walk C++'s `workspace_slots`." The merge was
    /// the fix; what survived it was a subtraction that had stopped meaning
    /// anything and a doc comment still describing the two lists.
    ///
    /// What DOES cross-check this is
    /// `workspace_parity::bytes_equals_the_sum_of_what_specs_would_allocate`,
    /// which sums the `TensorSpec`s an allocator would be handed. That is a
    /// second walk, over a different structure, and it can fail.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slots()
            .into_iter()
            .map(|(_, dtype, shape)| Self::slot_bytes(dtype, shape[0], shape[1]))
            .sum()
    }

    /// Bytes for one buffer, with negative extents floored at zero so a
    /// nonsense shape cannot wrap into an enormous figure.
    fn slot_bytes(dtype: DType, rows: i64, cols: i64) -> u64 {
        (rows.max(0) as u64)
            .saturating_mul(cols.max(0) as u64)
            .saturating_mul(dtype.size_bytes() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-0.6B, the first row of the parity grid.
    fn qwen3_0_6b() -> WorkspaceLayout {
        WorkspaceLayout::new(WorkspaceShape {
            hidden_size: 1024,
            vocab_size: 151_936,
            head_dim: 128,
            head_dim_kernel: 128,
            max_tokens: 2048,
            max_intermediate: 3072,
            max_hq: 16 * 128,
            max_hk: 8 * 128,
            max_output_rows: 64,
            max_mtp_draft_rows: 0,
        })
    }

    #[test]
    fn the_two_formerly_unbudgeted_buffers_are_worth_asserting_over() {
        let l = qwen3_0_6b();
        let s = l.shape();
        let declared = (s.max_tokens * (s.hidden_size + s.max_intermediate) * 2) as u64;
        let row0 = (s.vocab_size * 2) as u64;
        let without: u64 = l
            .slots()
            .into_iter()
            .filter(|(slot, _, _)| !matches!(slot, Slot::DeclaredValues | Slot::MtpRow0Save))
            .map(|(_, dtype, shape)| {
                (shape[0] as u64) * (shape[1] as u64) * dtype.size_bytes() as u64
            })
            .sum();
        assert_eq!(l.bytes() - without, declared + row0);
    }

    #[test]
    fn the_padded_branch_adds_four_buffers_and_nothing_else() {
        let mut shape = *qwen3_0_6b().shape();
        let packed = WorkspaceLayout::new(shape).slots().len();
        shape.head_dim = 96;
        shape.head_dim_kernel = 128;
        let padded = WorkspaceLayout::new(shape);
        assert_eq!(padded.slots().len(), packed + 4);
        assert!(padded.needs_padded_qkv());
    }

    #[test]
    fn draft_rows_extend_logits_without_moving_the_target_rows() {
        let mut shape = *qwen3_0_6b().shape();
        shape.max_mtp_draft_rows = 192;
        let l = WorkspaceLayout::new(shape);
        assert_eq!(l.mtp_draft_row_base(), shape.max_tokens);
        assert_eq!(l.logits_rows(), shape.max_tokens + 192);
    }

    #[test]
    fn a_negative_draft_reserve_is_clamped_not_subtracted() {
        let mut shape = *qwen3_0_6b().shape();
        shape.max_mtp_draft_rows = -8;
        let l = WorkspaceLayout::new(shape);
        assert_eq!(l.logits_rows(), shape.max_tokens);
    }

    #[test]
    fn every_slot_the_layout_names_is_produced_exactly_once() {
        let mut shape = *qwen3_0_6b().shape();
        shape.head_dim = 96;
        shape.head_dim_kernel = 128;
        let slots: Vec<Slot> = WorkspaceLayout::new(shape)
            .slots()
            .into_iter()
            .map(|(s, _, _)| s)
            .collect();
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), slots.len(), "a slot appears twice");
        assert_eq!(slots.len(), 25, "the padded layout has 25 buffers");
    }
}
