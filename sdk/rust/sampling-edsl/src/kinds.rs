//! Canonical sampler-kind taxonomy + the recognizer ladder (Task #8).
//!
//! [`CanonicalKind`] is the SDK-side classification of a sampler — the
//! host-derivable routing tag the de-hardwiring recognizer dispatches on
//! (dedicated kernel / baked IR / custom JIT). It is **routing-only**: it never
//! changes the token stream.
//!
//! [`infer_kind`] is the frozen recognizer ladder, transcribed **verbatim** from
//! the runtime's landed `infer_sampler_kind` (`c6865bac`) — the single source of
//! truth. The SDK sugar stamps [`CanonicalKind`] from this same ladder
//! ([`crate::canonical_kind`]); a drift-guard test asserts the SDK kind equals
//! the runtime-inferred kind for every standard sampler, so the recognizer and
//! the SDK can never silently diverge (the neutrality proof).

/// The canonical kind of a sampler. The six standard kinds mirror the runtime's
/// `StandardSamplerKind`; [`Custom`](CanonicalKind::Custom) tags a program that
/// is **not** a recognized standard sampler (a `Graph`-authored EDSL program) —
/// dispatched to the custom-JIT path, orthogonal to param-inference.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CanonicalKind {
    /// Greedy `argmax` (temperature ≤ 0, any filters).
    Argmax,
    /// Temperature-scaled multinomial (no truncation filter).
    Temperature,
    /// Min-p truncation.
    MinP,
    /// Top-k truncation.
    TopK,
    /// Top-p (nucleus) truncation.
    TopP,
    /// Combined top-k ∩ top-p.
    TopKTopP,
    /// Not a recognized standard sampler (a custom EDSL program) → custom JIT.
    #[default]
    Custom,
}

/// The frozen recognizer ladder — a verbatim transcription of the runtime's
/// `infer_sampler_kind` (`c6865bac`). Precedence is **first-match**:
///
/// 1. `temperature <= 0` → [`Argmax`](CanonicalKind::Argmax) (greedy ≡ argmax,
///    *any* k/p/min_p — the max-logit token survives every filter).
/// 2. `top_k>0 && 0<top_p<1` → [`TopKTopP`](CanonicalKind::TopKTopP) (the
///    combined arm, before the standalone arms — no filter dropped).
/// 3. `top_k>0` → [`TopK`](CanonicalKind::TopK).
/// 4. `0<top_p<1` → [`TopP`](CanonicalKind::TopP).
/// 5. `min_p>0` → [`MinP`](CanonicalKind::MinP).
/// 6. else (`T>0`, no filters) → [`Temperature`](CanonicalKind::Temperature).
///
/// Never returns [`Custom`](CanonicalKind::Custom): custom is orthogonal (a
/// program-bytecode-present signal handled before param-inference runs).
pub fn infer_kind(temperature: f32, top_k: u32, top_p: f32, min_p: f32) -> CanonicalKind {
    if temperature <= 0.0 {
        return CanonicalKind::Argmax;
    }
    let has_top_k = top_k > 0;
    let has_top_p = top_p > 0.0 && top_p < 1.0;
    let has_min_p = min_p > 0.0;
    if has_top_k && has_top_p {
        CanonicalKind::TopKTopP
    } else if has_top_k {
        CanonicalKind::TopK
    } else if has_top_p {
        CanonicalKind::TopP
    } else if has_min_p {
        CanonicalKind::MinP
    } else {
        CanonicalKind::Temperature
    }
}
