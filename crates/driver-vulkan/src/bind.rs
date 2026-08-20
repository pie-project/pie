//! This backend's half of `kernels::bind`.
//!
//! A routine states its arguments and, beside them, a `sources` column saying
//! where each one comes from: a slot of the statement, a scalar of it, a fact
//! of the fire, an arithmetic of those. Reading that column is the same work
//! on every backend, so `kernels::bind` does it once and asks a driver only
//! the questions a driver can answer -- which buffer is input 2, what is this
//! fire's row count, where is the KV cache for this layer.
//!
//! [`Held`] is this crate's answer to those questions. Everything it needs is
//! already in [`Handles`] and [`Facts`]; what it adds is the mapping from a
//! `kernels::keys` KEY to the thing this driver holds under that name.
//!
//! # Why a shared binder and not an arm
//!
//! Because the column was already stated and reading it twice is how the two
//! readings part. `driver-metal` and `driver-wgpu` each kept a hand-written
//! arm per crossed kernel -- a hundred functions restating an order their own
//! signatures already gave -- and when wgpu's were finally read against the
//! column before deletion, three had silently drifted: an arena operand bound
//! where a packed run belonged, an input read for a weight, and seven
//! arguments handed to a shader with six bindings. None was caught by a test,
//! because the arm WAS the test's idea of what the kernel took.
//!
//! This backend crosses the same kernels through the same signatures. It gets
//! the same binder.

use kernels::bind::{Answer, Holds};
use kernels::Ty;
use kernels::routine::Refusal;
use kernels_vulkan::routine::ArgValue;

use crate::hold::{Facts, Handles};
use crate::binding::{FireNumber, FireTable};

/// A statement and a fire, together, as the shared binder asks them.
pub struct Held<'a, 'h, 'o> {
    /// The statement's operands and this driver's pools.
    pub o: &'o mut Handles<'a, 'h>,
    /// The fire's geometry.
    pub f: Facts,
}

/// Bind one routine's arguments through the shared binder.
///
/// # Errors
///
/// Whatever the binder refuses with: an operand the statement does not carry,
/// a fact this backend does not answer, a carrier a value does not fit.
pub fn bind(
    args: &[Ty],
    sources: &[Option<kernels::Source>],
    o: &mut Handles<'_, '_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let mut held = Held { o, f };
    kernels::bind::bind::<ArgValue, _>(args, sources, &mut held)
}

/// ONE value, for a body that ASKS rather than a column that declares.
///
/// The same resolver, entered at one argument instead of a list: `ctx.ask::<C,
/// keys::X>()` resolves the key's own `Source`, `ctx.params()` the staged
/// block, and `ctx.absent()` a null. Nothing new answers — what changed is
/// only where the question is asked from.
///
/// # Errors
///
/// [`Refusal::Unstated`] for a fact this backend does not answer, and whatever
/// the fact's own absence means otherwise.
pub fn one(
    ty: Ty,
    source: kernels::Source,
    o: &mut Handles<'_, '_>,
    f: Facts,
) -> Result<ArgValue, Refusal> {
    let mut held = Held { o, f };
    kernels::bind::one::<ArgValue, _>(ty, source, &mut held)
}

impl Holds for Held<'_, '_, '_> {
    fn input(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.input(n)
    }

    fn output(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.output(n)
    }

    fn output_read(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.output_read(n)
    }

    // THE RECTANGLE BESIDE THE HANDLE. Without these two the shared binder's
    // `shaped` falls back to a width of zero for every operand, and the first
    // body that reads `x.width` refuses `Empty` -- which is what every fire
    // this driver planned did, at its first `embed_gather`.
    fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.in_width(n)
    }

    fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.out_width(n)
    }

    fn weight(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.weight(n)
    }

    fn params_block(&mut self) -> u32 {
        // VULKAN HAS NO PARAMS BLOCK to bind. Its scalars ride a push
        // constant range the encoder packs from the bound list, so the block
        // a metal or wgpu signature names as an operand is not an operand
        // here at all -- and a routine that names one gets this driver's
        // unbound placeholder, which is what `Handles::params_block` has
        // always returned.
        self.o.params_block()
    }

    fn param(&self, n: usize) -> Result<i32, Refusal> {
        self.o.param(n)
    }

    fn param_f32(&self, n: usize) -> Result<f32, Refusal> {
        self.o.param_f32(n)
    }

    fn null(&mut self) -> u32 {
        self.o.unbound()
    }

    fn fact(&mut self, key: &'static str) -> Option<Result<Answer, Refusal>> {
        Some(named(key, self.o, self.f))
    }
}

/// WHERE a fact comes from, decided without asking for it.
///
/// The split is what lets this backend's column be measured at all. A
/// `Handles` needs `Bound`s, which need real `vk::Buffer`s, which need a
/// device -- so a test that ran the binder could only run where there is a
/// GPU, and the question "does this driver answer every source its own
/// kernels name" has nothing to do with a GPU. [`whence`] answers that half,
/// [`named`] is written in terms of it, and there is still only one spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Whence {
    /// One of the fire's tables, staged per run.
    Table(FireTable),
    /// This layer's KV cache: keys, or values.
    Kv(bool),
    /// A per-layer slab, by the name a signature gives it.
    Slab(&'static str),
    /// One of the fire's own numbers, which reaches a kernel as a scalar.
    Pooled(FireNumber),
    /// The fire's geometry, which [`Facts`] already carries.
    Geometry(i32),
    /// The split decode's scratch, which is a table like any other but is
    /// asked for through a key only this backend answers.
    Partials,
    /// How many ways this fire's decode splits, which is a JUDGEMENT rather
    /// than a lookup -- see [`Whence::Splits`]'s arm in `named`.
    Splits,
}

/// Where the thing a key names comes from, or `None` if this backend has
/// never heard of the key.
#[must_use]
pub fn whence(key: &str, f: Facts) -> Option<Whence> {
    use kernels::keys::{self, Fact};

    if let Some(which) = table_of(key) {
        return Some(Whence::Table(which));
    }
    if key == keys::KvKeys::KEY {
        return Some(Whence::Kv(false));
    }
    if key == keys::KvValues::KEY {
        return Some(Whence::Kv(true));
    }
    for slab in SLABS {
        if key == slab {
            return Some(Whence::Slab(slab));
        }
    }
    for (k, which) in POOLED {
        if key == k {
            return Some(Whence::Pooled(which));
        }
    }
    if key == keys::AttnPartials::KEY {
        return Some(Whence::Partials);
    }
    if key == keys::AttnSplits::KEY {
        return Some(Whence::Splits);
    }
    geometry(key, f).map(Whence::Geometry)
}

/// A fact, by the key the signature names.
///
/// The tables and the cache come back as handles; the geometry comes back as
/// numbers off [`Facts`]. Both are this backend's to answer, which is what
/// `Provenance::Env` said all along -- this says WHICH.
fn named(key: &'static str, o: &mut Handles<'_, '_>, f: Facts) -> Result<Answer, Refusal> {
    let Some(from) = whence(key, f) else {
        return Err(Refusal::Unstated {
            what: "a fact this backend does not answer",
        });
    };
    Ok(match from {
        Whence::Table(which) => Answer::Handle(o.table(which)?),
        Whence::Kv(values) => Answer::Handle(o.kv(f.layer, values)?),
        Whence::Slab(name) => Answer::Handle(o.slab(f.layer, name)?),
        // `Wide` rather than `Number` because a stride is a byte count and
        // the signature says whether it fits: `Usize` takes it whole and a
        // narrower carrier refuses a deployment large enough to overflow,
        // which is the honest answer where a cast is a silent truncation.
        Whence::Pooled(which) => Answer::Wide(u64::from(o.number(which)?)),
        Whence::Geometry(n) => Answer::Number(n),
        Whence::Partials => Answer::Handle(o.table(FireTable::AttnPartials)?),
        // NOT A LOOKUP, and that is why it lives here rather than in a
        // signature. How many ways to fold a decode's key range is a
        // judgement about THIS fire -- its history depth against its head
        // count and row count -- and a signature is fire-invariant, so it can
        // name the number but cannot compute it. `contiguous_pool` is the
        // same shape on `driver-wgpu`: a rule that reads like an arm, is not
        // one, and belongs in the half of binding that is the backend's.
        //
        // A fire with no partials buffer or no history bucket answers ONE,
        // the single-pass path, because zero splits is not a thing.
        Whence::Splits => Answer::Number(o.decode_splits(f)),
    })
}

/// The slabs a key may name, which the driver holds per layer.
const SLABS: [&str; 3] = ["conv_state", "new_conv_state", "recurrent_state"];

/// The pool's numbers, which reach a kernel as scalars rather than addresses.
///
/// `KvHeadStride` and `KvSeqStride` are NOT guarded against a paged pool the
/// way `driver-wgpu`'s are, because this driver's `Shape::number` returns
/// `None` for them over a paged pool rather than deriving one -- the refusal
/// is at the source, so a second one here would be a second spelling of it.
const POOLED: [(&str, FireNumber); 4] = [
    (
        <kernels::keys::KvHeadStride as kernels::keys::Fact>::KEY,
        FireNumber::KvHeadStride,
    ),
    (
        <kernels::keys::KvSeqStride as kernels::keys::Fact>::KEY,
        FireNumber::KvSeqStride,
    ),
    (
        <kernels::keys::KvPageSize as kernels::keys::Fact>::KEY,
        FireNumber::KvPageSize,
    ),
    // The attention mask's row pitch. Not the pool's, and this driver stages
    // a real mask, so the number is the width it made each row -- which is
    // why all three backends' `sdpa_paged_*` rows now ask the fire for it.
    (
        <kernels::keys::AttentionMaskStride as kernels::keys::Fact>::KEY,
        FireNumber::AttentionMaskStride,
    ),
];

/// The fire's GEOMETRY, by key. `None` for a key that is not one of the
/// fire's numbers, which is a refusal at the caller rather than here.
fn geometry(key: &str, f: Facts) -> Option<i32> {
    use kernels::keys::{self, Fact};

    Some(match key {
        _ if key == keys::Rows::KEY => f.rows.cast_signed(),
        _ if key == keys::RequestCount::KEY => f.requests.cast_signed(),
        _ if key == keys::Width::KEY => f.width.cast_signed(),
        _ if key == keys::InWidth::KEY => f.in_width.cast_signed(),
        _ if key == keys::NumQHeads::KEY => f.q_heads.cast_signed(),
        _ if key == keys::NumKvHeads::KEY => f.kv_heads.cast_signed(),
        _ if key == keys::HeadDim::KEY => f.head_dim.cast_signed(),
        _ if key == keys::NumExperts::KEY => f.n_experts.cast_signed(),
        _ if key == keys::ExpertsPerToken::KEY => f.experts_per_token.cast_signed(),
        _ if key == keys::RotaryWidth::KEY => f.rotary_dims.cast_signed(),
        _ if key == keys::VHeads::KEY => f.v_heads.cast_signed(),
        _ if key == keys::VDim::KEY => f.v_dim.cast_signed(),
        _ if key == keys::QuantGroup::KEY => f.group.cast_signed(),
        _ if key == keys::QuantBits::KEY => f.bits.cast_signed(),
        // The GEMM tile the SYMBOL spells, which is why it may be absent: a
        // matvec's entrypoint names no tile, and a signature that asks for
        // one over a symbol that does not spell one is asking about a
        // multiply that is not happening.
        _ if key == keys::TileM::KEY => f.tile?.0.cast_signed(),
        _ if key == keys::TileN::KEY => f.tile?.1.cast_signed(),
        _ => return None,
    })
}

/// The fire table a key names, if it names one.
fn table_of(key: &str) -> Option<FireTable> {
    use kernels::keys::{self, Fact};

    Some(match key {
        _ if key == keys::TokenIds::KEY => FireTable::TokenIds,
        _ if key == keys::Positions::KEY => FireTable::Positions,
        _ if key == keys::RequestOfToken::KEY => FireTable::RequestOfToken,
        _ if key == keys::KvPageIndices::KEY => FireTable::KvPageIndices,
        _ if key == keys::KvPageIndptr::KEY => FireTable::KvPageIndptr,
        _ if key == keys::AttentionMask::KEY => FireTable::AttentionMask,
        _ if key == keys::AttentionMaskEnabled::KEY => FireTable::AttentionMaskEnabled,
        _ if key == keys::KvWritePage::KEY => FireTable::KvWritePage,
        _ if key == keys::KvWriteOffset::KEY => FireTable::KvWriteOffset,
        _ if key == keys::RopeFrequencies::KEY => FireTable::RopeFrequencies,
        _ if key == keys::SamplingIndices::KEY => FireTable::SamplingIndices,
        _ => return None,
    })
}

#[cfg(test)]
// The crate denies `print_stdout` and means it -- a driver that prints is a
// driver whose output is somebody's log. A TEST that prints is how the counts
// below are read, and the two rules do not conflict once said apart.
#[allow(clippy::print_stdout, reason = "these tests report counts to be read")]
mod tests {
    use super::{Facts, whence};

    /// A fire wide enough that nothing refuses for want of a number.
    ///
    /// Every field distinct so a value arriving from the wrong one is visible
    /// rather than accidentally equal, and none of them zero: a group size of
    /// zero is `Refusal::Empty` and a zero axis makes a head count a division
    /// by it.
    pub(super) fn facts_for_test() -> Facts {
        Facts {
            rows: 4,
            width: 64,
            in_width: 48,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 16,
            rotary_dims: 12,
            n_experts: 6,
            experts_per_token: 3,
            group: 32,
            bits: 4,
            layer: 1,
            requests: 2,
            v_heads: 4,
            v_dim: 24,
            tile: Some((32, 64)),
        }
    }

    /// Every fact this backend's own kernels name is one it answers.
    ///
    /// The wgpu twin of this runs the whole binder and compares bound
    /// buffers. This one cannot: a `Handles` holds `Bound`s over real
    /// `vk::Buffer`s, so building one needs a device, and whether this driver
    /// KNOWS a key has nothing to do with whether a GPU is present. So it
    /// asks `whence`, which is the half `named` is written in terms of.
    ///
    /// The slot half -- input 2, output 0, the weights -- needs no gate here:
    /// the shared binder reads those the same way on three backends and
    /// `kernels`' own tests cover it.
    #[test]
    fn every_fact_this_backend_s_kernels_name_is_one_it_answers() {
        let f = facts_for_test();
        let mut unanswered: Vec<String> = Vec::new();
        let mut asked = 0usize;
        let mut whole = 0usize;
        let mut routines = 0usize;

        for routine in kernels_vulkan::routines() {
            routines += 1;
            let mut bad = 0usize;
            for (at, source) in routine.sources.iter().enumerate() {
                for key in keys_of(source.as_ref()) {
                    asked += 1;
                    if whence(key, f).is_none() {
                        bad += 1;
                        unanswered.push(format!("  {}[{at}]: {key}", routine.name));
                    }
                }
            }
            if bad == 0 {
                whole += 1;
            }
        }

        println!("routines whose facts are all answered: {whole} of {routines}");
        println!("facts named but not answered: {}", unanswered.len());
        for line in &unanswered {
            println!("{line}");
        }
        // The denominator. A column read as empty satisfies every line above,
        // and a `sources` accessor that changed shape is exactly how that
        // happens.
        assert!(
            asked > 200,
            "{asked} facts read off this backend's column -- the sweep found \
             almost nothing, so the emptiness below is the reader's and not \
             the column's"
        );
        // WHICH routines this backend cannot bind, named rather than
        // counted. Every one of them wants the recurrent SLOT table, and
        // this driver does not serve recurrent state at all: `frames.rs`
        // refuses a plan carrying `rs_slot_ids`, and
        // `engine/src/driver/backend/vulkan.rs` says so twice in prose. The
        // refusal `whence` produces is therefore the right answer and not a
        // hole -- the wrong answer would be a handle to something else.
        //
        // Named, because "five" would go on being true if a sixth routine
        // arrived and one of these was ported.
        assert_eq!(
            unanswered,
            vec![
                "  gdn_core_recurrent_prefill[6]: recurrent_slots",
                "  gdn_core_recurrent_slotted[11]: recurrent_slots",
                "  gdn_core_slotted[12]: recurrent_slots",
                "  gdn_prep_prefill[13]: recurrent_slots",
                "  gdn_prep_slotted[13]: recurrent_slots",
            ],
            "a different set of this backend's kernels names a fact it does \
             not answer. If one LEFT, this driver learned to stage the \
             recurrent slot table and the list comes down with it; if one \
             ARRIVED, a signature started naming a fact `whence` has never \
             heard of.",
        );
    }

    /// Every key a source names, following arithmetic into both sides.
    fn keys_of(source: Option<&kernels::Source>) -> Vec<&'static str> {
        let mut out = Vec::new();
        walk(source, &mut out);
        out
    }

    fn walk(source: Option<&kernels::Source>, out: &mut Vec<&'static str>) {
        match source {
            Some(kernels::Source::Named(key)) => out.push(key),
            Some(
                kernels::Source::Times(a, b)
                | kernels::Source::Over(a, b)
                | kernels::Source::Or(a, b),
            ) => {
                walk(Some(a), out);
                walk(Some(b), out);
            }
            _ => {}
        }
    }
}
