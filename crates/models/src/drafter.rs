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
