//! **BLOCK DRAFTERS THAT BRING THEIR OWN ARCHITECTURE**, shared across the
//! families that carry them.
//!
//! A chained draft head (an MTP or Eagle head) reuses its trunk's block —
//! qwen_3's gated attention, dsv4's MLA, gemma's own — so it lives beside
//! that trunk. A block drafter does not: DFlash ships five plain decoder
//! layers of its own geometry, fed by a fusion of tapped trunk hidden states
//! and read out through the target's `lm_head`, and nothing in it knows which
//! trunk it is tapping. What it needs from the family is exactly four hooks,
//! and every family that carries one spells the same four:
//!
//! 1. **split** — the draft rows leave the residual stream before the trunk's
//!    first layer (`y.split(&block_draft)`), so a fire of nothing but draft
//!    rows pays for no trunk layer;
//! 2. **tap** — inside the family's layer loop, [`dflash::DFlash::tap`] fuses
//!    the hidden state at each tapped layer AS IT PASSES (the residual stream
//!    is one aliased buffer; a handle held across layers reads the wrong one);
//! 3. **arm + merge** — [`dflash::DFlash::arm`] writes the drafter's context
//!    into its kv rows and runs the block; the family merges the block rows
//!    back in front of its one `lm_head`;
//! 4. **readout** — [`dflash::DFlash::plant_readout`] plants the head's
//!    proposals on the `mtp.drafts` seam (v1's argmax, v2's selector walk).
//!
//! Plus the two declarations: [`dflash::DFlash::declare_caches`] registers
//! the drafter's kv rows in a page-id space the family names (a space admits
//! rows of any plane width, so the drafter's 8 × 128 rides beside a 4 × 256
//! trunk), and [`dflash::DFlash::bind_aux`] binds its `aux.*` planes at
//! import. The family keeps: its `Facts` bit for `block_draft` (bit positions
//! are the family's), its `Recipe` (which head an artifact carries is the
//! family's choice), and its dtype policy.
//!
//! The engine side is already family-blind: `set-drafting-block` marks the
//! lanes, `mtp.drafts` carries the proposals, `--aux` prefixes the planes.

pub mod dflash;

/// **A PUBLISHED HEAD FOR A PUBLISHED TARGET** — the pairing an operator
/// would otherwise look up by hand: which repository carries the drafter
/// trained against which checkpoint, and which catalog row reads the two
/// together. `pie model import <target> --drafter <name>` resolves through
/// this table; so does `[model] drafter = "<name>"` when `[model] model`
/// names the target repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Published {
    /// The target checkpoint's repository id, as `pie model import` takes it.
    pub target: &'static str,
    /// The drafter's repository id — the `--aux` overlay.
    pub head: &'static str,
    /// The drafter's short name, what `--drafter` and `[model] drafter` say.
    pub drafter: &'static str,
    /// The catalog row that reads target and head together.
    pub sku: &'static str,
}

/// Every head this build knows how to overlay, by target.
pub const PUBLISHED: &[Published] = &[
    Published {
        target: "mlx-community/Qwen3.6-27B-4bit",
        head: "z-lab/Qwen3.6-27B-DFlash",
        drafter: "dflash",
        sku: "qwen36-27b-dflash-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.8-27B-4bit",
        head: "z-lab/Qwen3.8-27B-DFlash2",
        drafter: "dflash2",
        sku: "qwen38-27b-dflash2-u4g64-kv-bf16",
    },
];

/// The published head named `drafter` for `target`, if this build knows one.
/// `target` is matched as a repository id, case-insensitively, with or
/// without the `--` spelling a store directory uses for `/`.
#[must_use]
pub fn published(target: &str, drafter: &str) -> Option<&'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    PUBLISHED.iter().find(|p| {
        p.drafter.eq_ignore_ascii_case(drafter) && p.target.to_ascii_lowercase() == wanted
    })
}

/// Every published head this build knows for `target`.
pub fn published_for(target: &str) -> impl Iterator<Item = &'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    PUBLISHED.iter().filter(move |p| p.target.to_ascii_lowercase() == wanted)
}

