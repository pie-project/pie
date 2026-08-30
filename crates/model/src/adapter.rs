//! **THE CORRECTION AXIS, SAID ONCE FOR EVERY FAMILY** (alto adapter §3, palo
//! design §8, campaign A-6).
//!
//! Six family texts carry the same seam and it is the same seam in all six: a
//! pair of registered banks per layer, a `has_adapter` fact bit, and one
//! `linear.lora_correct` over the window that bit cuts. What differs between
//! families is only WHERE the site is — which value is the mixer's input and
//! which is its output after the collective — and that is the one thing a text
//! has to say for itself. Everything else is here, because six copies of a
//! shape declaration are six places for a rank to be wrong in.
//!
//! # What is NOT here, and why
//!
//! The numbers. [`Adapters`] is a struct and not a constant: how many adapters
//! a load seats and how wide their waist is are a DEPLOYMENT's ceiling written
//! where a shape has to be written, and what that ceiling costs is a fact
//! about the family that pays it (twelve mebibytes against qwen35-d0.8b's
//! 1.40 GiB is not the same sentence as four hundred against kimi's terabyte).
//! Each family states its own `ADAPTERS` beside its own arithmetic.

use model_dsl::{Dtype, Weight};

/// **THE BUDGET IS THE SHAPE** (palo design §8, decision 17).
///
/// How many adapters a load can hold and how wide each one's waist is are
/// declared in the MODEL TEXT, because they are the leading axes of two
/// weights and shapes are the text's. `Budget::max_adapters` is what a
/// deployment asks to be able to register, and `model_compiler::compile`
/// refuses a load whose ask is bigger than what these numbers seat — one
/// refusal, at the door, instead of a capacity discovered at a registration.
///
/// **RANK DIVERSITY IS BUCKETED BY BANK, NOT BY A BRANCH.** An adapter trained
/// at a lower rank is registered zero-padded into `rank` — which is exact, a
/// zero row of `A` contributing a zero to the waist — and a deployment whose
/// adapters spread widely across ranks declares a second family SKU with a
/// second `rank` rather than a runtime rank table nothing but the padding
/// would read. That is design §8's "rank-bucketed grouped GEMM" read
/// literally: the buckets are banks.
#[derive(Clone, Copy, Debug)]
pub struct Adapters {
    /// How many adapters the bank's first axis seats.
    pub slots: u32,
    /// The waist every adapter of this bank is padded to.
    pub rank: u32,
}

/// One correction site's two banks, named under `prefix`:
/// `{prefix}.lora_a` is `[slots, rank, hidden]` and `{prefix}.lora_b` is
/// `[slots, hidden, rank]`.
///
/// **REGISTERED, NOT LANDED.** No checkpoint publishes either of these and the
/// loader must not demand them: they are reserved at load and zeroed, and a
/// zeroed `A` is the identity — so a fire through an unwritten row of the bank
/// says exactly what the base model says. That is what makes A-1's zero-scale
/// parity a property of the declaration rather than of a code path.
///
/// **THE ORIENTATIONS ARE THE ENGINE'S STATUTE AND NOT A PREFERENCE**
/// (adapter §6.3). `A` is rank-major and `B` is out-major — HF's own
/// orientation for `lora_B` — and `engine_cuda::adapter::Role` refuses a plane
/// filled the other way round by name. Stating them here means a family cannot
/// declare a transposed bank that loads and then answers noise.
///
/// **AND `dense`, NOT `w`.** A correction is written by the HOST into a
/// reserved plane; it is not a checkpoint bank and has no group to quantize
/// against, so it is declared in the plan's compute element the way every norm
/// in these texts is (`crate::dense`).
///
/// **THE NAME CARRIES NO SITE, AND THAT IS A KNOWN LIMIT** (A-2's report).
/// `engine_cuda::blob::role_of` reads a bank name as `layer.{n}.{role}` and
/// matches `role` against `lora_a`/`lora_b` literally, so a site-tagged name
/// (`layer.{n}.mixer.lora_a`) parses as its own role at layer zero and every
/// shared-adapter bind refuses. Which site a text corrects is therefore the
/// text's alone, and a guest that binds an adapter gets the one site the
/// family below chose. Widening it is an engine-cuda edit (`role_of`,
/// `layer_of`, and the manifest's `role` key), not a model-text one.
#[must_use]
pub fn banks(prefix: &str, a: Adapters, hidden: u64, dense: Dtype) -> (Weight, Weight) {
    let slots = u64::from(a.slots);
    let rank = u64::from(a.rank);
    (
        Weight::sym(format!("{prefix}.lora_a"), [slots, rank, hidden], dense).registered(),
        Weight::sym(format!("{prefix}.lora_b"), [slots, hidden, rank], dense).registered(),
    )
}
