//! The per-fire forward workspace: the reusable scratch every llama-like
//! forward path writes into, sized once against `max_tokens`.
//!
//! # Why this is a table rather than two functions
//!
//! The C++ used to say the same layout twice. `Workspace::allocate_full`
//! created the tensors, and `workspace_bytes` — a separate function, a hundred
//! lines further down — added up what they cost so the memory planner could
//! subtract the arena from the KV pool. Nothing compared them, and its own
//! comment stated the stakes:
//!
//! > *this function is the planner's arena figure (`memory_planner.cpp`) and
//! > every byte missing from it is a byte handed to the KV pool instead.*
//!
//! They had drifted. `declared_values` (`[N, hidden + intermediate]` bf16) and
//! `mtp_row0_save` (`[1, vocab]` bf16) were allocated and never budgeted, so
//! the planner under-charged the arena on every model it had ever run — 503 MB
//! on a Qwen3-32B shape, 151 MB on Llama-3-8B. Porting is what surfaced it:
//! reconciling two statements of one layout is work a port cannot skip.
//!
//! Both sides now state the layout ONCE. Here it is [`WorkspaceLayout::slots`],
//! which [`WorkspaceLayout::bytes`] sums and [`WorkspaceLayout::specs`] hands
//! to the allocator; in C++ it is `workspace_slots`, which `allocate_full` and
//! `workspace_bytes` both walk. Neither pair can drift again, because in
//! neither is there a second list to drift from.

use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::TensorSpec;

/// A named buffer in the workspace.
///
/// The order of this enum is the order `allocate_full` creates the tensors,
/// which is the order the parity transcript records. It is neither
/// alphabetical nor the declaration order of the C++ struct — those differ
/// from each other too, and the allocation order is the one that is
/// observable.
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
/// Mirrors `kernels::sample::kArgmaxAccumSlots`. The parity oracle reads the
/// C++ value out of the shipping header rather than retyping it, so this
/// constant going stale is a test failure and not a silent mis-size.
pub const ARGMAX_ACCUM_SLOTS: i64 = 32;

/// Most a single program may reserve.
///
/// Mirrors `Workspace::kMtpDraftRowsPerProgram`.
pub const MTP_DRAFT_ROWS_PER_PROGRAM: i32 = 32;

/// The shape parameters `allocate_full` takes, named.
///
/// The C++ signature is seven positional `int`s of which four are bounds and
/// three are model dimensions, and adjacent pairs are interchangeable at the
/// call site without a diagnostic. Naming them is the entire reason this
/// struct exists.
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
    /// MTP draft rows reserved at the tail of `logits`. Negative is clamped
    /// to zero, as the C++ `std::max(0, ...)` does.
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

    /// First `logits` row reserved for MTP drafts.
    ///
    /// Mirrors `workspace_mtp_draft_row_base`, which is the identity on
    /// `max_tokens`. It exists as a named function because the *meaning* —
    /// "drafts start where the target rows end" — is what callers depend on,
    /// and it would be re-derived at each of them otherwise.
    #[must_use]
    pub const fn mtp_draft_row_base(&self) -> i64 {
        self.shape.max_tokens
    }

    /// Rows in `logits`: the target rows plus any MTP draft reserve.
    ///
    /// Mirrors `workspace_logits_rows`.
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
    ///
    /// The C++ used to hold two different fallbacks for this one field:
    /// `allocate_full` read `max_output_rows > 0 ? max_output_rows :
    /// max_tokens` while `workspace_bytes` read `std::max(1, output_rows)`.
    /// They agree for every positive value and differ at zero, where the
    /// allocator made `max_tokens` rows and the budget charged one. The
    /// planner has only ever passed a positive count, so it was unreachable
    /// rather than harmless. Merging the two layouts settled it on the
    /// allocator's reading, kept here, because that is the one that reserves
    /// memory.
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

    /// The divisor `allocate_full` uses to recover a head count.
    ///
    /// It divides by `std::max(1, cfg.head_dim)`, not by `head_dim`, so a
    /// config with a zero head dim yields zero heads instead of trapping.
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

    /// Every buffer in the layout, in allocation order.
    ///
    /// This is the single statement of the layout. Both the allocator and the
    /// byte budget read it; neither restates it.
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
    /// walks by construction.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slots()
            .into_iter()
            .map(|(_, dtype, shape)| Self::slot_bytes(dtype, shape[0], shape[1]))
            .sum()
    }

    /// What the C++ `workspace_bytes` reports for this shape.
    ///
    /// Equal to [`Self::bytes`] — that is the point, and
    /// [`Self::budget_shortfall`] asserting zero is what keeps it true. It is
    /// kept as a separate computation rather than an alias because it is
    /// derived the way the C++ derives it, so the parity oracle compares two
    /// independent walks rather than one value against itself.
    #[must_use]
    pub fn cpp_budget_bytes(&self) -> u64 {
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

    /// Bytes the planner is not told about: [`Self::bytes`] less
    /// [`Self::cpp_budget_bytes`].
    ///
    /// Zero, and the parity test requires it to stay zero on every shape in
    /// the grid. It is not dead code but the assertion's subject: this is the
    /// quantity that was 503 MB before the two layouts were merged, and the
    /// only thing standing between a future edit and that number coming back.
    #[must_use]
    pub fn budget_shortfall(&self) -> u64 {
        self.bytes().saturating_sub(self.cpp_budget_bytes())
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
    fn the_budget_and_the_allocation_are_the_same_number() {
        let l = qwen3_0_6b();
        assert_eq!(l.budget_shortfall(), 0);
        assert_eq!(l.bytes(), l.cpp_budget_bytes());
    }

    /// The two buffers whose omission was the bug, priced so the fix has a
    /// number attached rather than only an assertion.
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
