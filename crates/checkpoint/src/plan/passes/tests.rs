//! The pass pipeline's own checks.
//!
//! These test the passes against a hand-built plan rather than through a
//! contract, which is the point: a plan can be malformed in ways no contract
//! can express, and the validators exist for exactly those.


// ---------------------------------------------------------------------------
// The scratch region: what a slot may be shared with, and what it may not.
//
// This is the allocator whose mistakes are silent. Two operands placed at one
// offset while both are live is not a crash and not a refusal — it is a weight
// that loads and holds the other one's bytes. So the reuse rule is tested
// directly rather than inferred from a plan that happens not to provoke it.
// ---------------------------------------------------------------------------

