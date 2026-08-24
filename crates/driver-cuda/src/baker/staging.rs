//! The staging shim: the routines that keep their own `canon`.
//!
//! A `Call::Symbol` is a routine for which no honest delegation to a point
//! exists — the plane's `#[claims]` blocks name each one where it is
//! measured, `kernels-cuda/src/ssm.rs`'s family doc being the longest.
//! Those need STAGING: operands the statement does not carry, results it
//! does not state, resident objects it only names. That is what this file
//! is, and unlike the generated dispatch beside it this is NOT a generator's
//! output — a generator has nothing to read here, because the gap between
//! what the statement says and what the routine wants is exactly the thing no
//! declaration captures. Each of these dies when its routine is decomposed
//! into points that state their own operands.
//!
//! FOUR OF THE FIVE ARMS ARE GONE, and the two that left last left the same
//! way the first two did.
//!
//! W10 took the two `ssm` arms — the ones that needed the most staging: a
//! backwards reach through the plan for an operand the statement never
//! carried, three carved scratch columns for results it never stated, and
//! four column cuts into packed rows. Every one of those was the symptom of
//! a routine whose signature was not the point's, and both are claim bodies
//! now.
//!
//! `layout.embed` (R4a) died the second way an arm can: its staging was not
//! a rectangle but ONE NUMBER — the table's row count, read out of the
//! plan's weight shape because the point stated no such number. The
//! declaration states `vocab` now, so the number reaches the plane through
//! the statement like every other geometry.
//!
//! R4b took `attn::write_kv_to_pages` and
//! `attn::dispatch_attention_flashinfer_decode`, whose staging was of the
//! third kind: not operands the statement should have carried, but PLANE
//! STAGING no statement can carry — the fa2 decode schedule, the fire's
//! write origin, its query CSR, its row-validity plane. Those did not need
//! to move onto the operand column; they needed a door on `self`, which is
//! `Ctx::raised` keyed by the `Raise` each object declares and
//! `bind::views::FireViews` answering it.
//!
//! `Call::Symbol` stays a variant while any point in the tree keeps a
//! canon: cuda has two — `norm.res_blend` and `hc.collapse`, both argued in
//! `kernels/src/points.rs`, both waiting on something the floor does not
//! have (a `Vararg` mark; a producer for the head-gate logits no text
//! writes). A lane that carries one has to refuse BY NAME rather than
//! fault, and the `other` arm below is that refusal.

use kernels::routine::Refusal;
use model_ir::plan::Op;

use super::fire::Fire;

/// Fire one `Call::Symbol` through the routine it names.
pub(crate) fn symbol(f: &Fire<'_>, symbol: &str, op: &Op) -> Result<(), Refusal> {
    let _ = (f, op);
    Err(Refusal::Absent {
        what: Box::leak(
            format!("a staging shim for `{symbol}`; this driver states none").into_boxed_str(),
        ),
    })
}
