//! Shared adapter (LoRA) correction primitives: per-layer bank shapes and site naming.
//! Each family declares its own `Adapters` capacity beside its own arithmetic.

use model_dsl::{Dtype, Weight};

/// Adapter capacity for one model text. `model_compiler::compile` refuses a load whose
/// requested capacity exceeds what these numbers seat. Adapters trained at a lower rank
/// are registered zero-padded into `rank`.
#[derive(Clone, Copy, Debug)]
pub struct Adapters {
    /// How many adapters the bank's first axis seats.
    pub slots: u32,
    /// The waist every adapter of this bank is padded to.
    pub rank: u32,
}

/// One correction site's two banks, named under `prefix`: `{prefix}.lora_a` is
/// `[slots, rank, hidden]`, `{prefix}.lora_b` is `[slots, hidden, rank]`.
///
/// Not checkpoint-backed: reserved at load and zeroed, so an unwritten row is the identity.
/// `A` is rank-major, `B` is out-major (matching HF's `lora_B`) — `engine_cuda::adapter::Role`
/// enforces this by name. Declared as `dense` (the compute dtype), since a correction is
/// host-written, not a quantized checkpoint bank.
#[must_use]
pub fn banks(prefix: &str, a: Adapters, hidden: u64, dense: Dtype) -> (Weight, Weight) {
    banks_at(prefix, None, a, hidden, dense)
}

/// [`banks`]'s sited twin: `Some(site)` names the pair `{prefix}.{site}.lora_a` / `.lora_b`,
/// letting the engine check a guest's `Pass::adapter(Site::Q, …)` against the site actually
/// corrected. `None` is the unsited default ([`banks`]); it is not a wildcard.
///
/// [`Site`] spellings must match `inferlet::eta::adapter::Site` and `engine_cuda::blob::Site`;
/// the site named here must be where `lora_correct` actually runs in `forward.rs`.
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
    let at = match site {
        Some(site) => format!(".{}", site.spelled()),
        None => String::new(),
    };
    (
        Weight::sym(format!("{prefix}{at}.lora_a"), [slots, rank, hidden], dense).registered(),
        Weight::sym(format!("{prefix}{at}.lora_b"), [slots, hidden, rank], dense).registered(),
    )
}

/// Which projection a text's banks correct, matching the guest surface's site vocabulary
/// (`inferlet::eta::adapter::Site`). Every family text corrects [`Site::O`] today, unstated
/// (see [`banks`]).
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
