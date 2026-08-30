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
/// **AND THE NAME MAY NOW CARRY THE SITE** (alto next B3, closing A-2's
/// report). It could not until this wave: `engine_cuda::blob::role_of` read a
/// bank name as `layer.{n}.{role}` and matched `role` literally, so a
/// site-tagged name parsed as its own role at layer zero and every shared
/// bind refused — which meant a guest's `Site::Q` silently got whichever site
/// the family below chose. The parser now accepts an optional site segment
/// (`layer.{n}.{site}.{lora_a|lora_b}`) and the bind path refuses a guest that
/// asked for a site the banks do not declare, by name.
///
/// [`banks_at`] is how a text states one. **THIS FUNCTION IS THE UNSTATED
/// DEFAULT** and its names are what they have always been — the six family
/// texts call it and are byte-identical across this wave.
#[must_use]
pub fn banks(prefix: &str, a: Adapters, hidden: u64, dense: Dtype) -> (Weight, Weight) {
    banks_at(prefix, None, a, hidden, dense)
}

/// [`banks`]'s sited twin: the same pair, named at the correction site the
/// text says they correct.
///
/// `Some(site)` names them `{prefix}.{site}.lora_a` / `.lora_b`, which is what
/// lets the engine check a guest's `Pass::adapter(Site::Q, …)` against what
/// this text actually corrects instead of serving it the default site
/// unchecked. `None` is [`banks`] — the pre-B3 pair, unstated, meaning the
/// text's own one site — and it is not a wildcard: on a load whose banks name
/// sites, a bind that states none is refused like any other mismatch.
///
/// **THE SPELLINGS ARE THE CONTRACT** and they are the guest surface's:
/// [`Site`] is `inferlet::eta::adapter::Site` in snake case, and
/// `engine_cuda::blob::Site` is the same six words on the other side of the
/// name. Three copies of a vocabulary because a name crosses a crate boundary
/// as a string — the same reason `lora_a` is written here and matched there —
/// and a spelling the engine does not know is REFUSED rather than guessed at.
///
/// **A TEXT THAT STATES A SITE MUST MEAN IT.** The site named here is where
/// the family's `lora_correct` actually stands in `forward.rs`; naming a
/// second site's banks without correcting at it would declare a capacity that
/// answers nothing, which is a bank the engine will seat and no fire will
/// read.
#[must_use]
pub fn banks_at(
    prefix: &str,
    site: Option<Site>,
    a: Adapters,
    hidden: u64,
    dense: Dtype,
) -> (Weight, Weight) {
    let slots = u64::from(a.slots);
    let rank = u64::from(a.rank);
    // The unstated default is an EMPTY infix and not a word, which is what
    // makes `banks` byte-identical to the function it replaced: the names it
    // builds are `{prefix}.lora_a` and `{prefix}.lora_b`, character for
    // character what the six family texts have always declared.
    let at = match site {
        Some(site) => format!(".{}", site.spelled()),
        None => String::new(),
    };
    (
        Weight::sym(format!("{prefix}{at}.lora_a"), [slots, rank, hidden], dense).registered(),
        Weight::sym(format!("{prefix}{at}.lora_b"), [slots, hidden, rank], dense).registered(),
    )
}

/// **WHICH PROJECTION A TEXT'S BANKS CORRECT** — the guest surface's own site
/// vocabulary (`inferlet::eta::adapter::Site`), spelled as a name segment.
///
/// Six llama-like projection sites, which is the whole of what
/// `Pass::adapter` can ask for. Every family text corrects [`Site::O`] today
/// — the mixer's output after its collective — and states it by not stating
/// it (see [`banks`]); this enum is what a text uses when it grows a second
/// correction site and the two have to be told apart by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The mixer's output projection.
    O,
    /// The fused gate/up projection of the feed-forward sublayer.
    GateUp,
    /// Its down projection.
    Down,
}

impl Site {
    /// The one segment a bank name spells it with.
    #[must_use]
    pub const fn spelled(self) -> &'static str {
        match self {
            Site::Q => "q",
            Site::K => "k",
            Site::V => "v",
            Site::O => "o",
            Site::GateUp => "gate_up",
            Site::Down => "down",
        }
    }
}
