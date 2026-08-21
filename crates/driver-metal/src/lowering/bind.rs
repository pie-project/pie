//! What the FIRE answers, on this backend. The rest of binding is shared.
//!
//! `arm.rs` was ninety-one functions that each said, in Rust, where every
//! argument of one routine comes from. STAGES 2 through 6 of
//! `.wiki/kilimanjaro4.md` moved that same knowledge into the signatures
//! themselves, as the `sources` column every `KernelFn` derives, and gated
//! each move against the arm it was copied from. Reading that column is
//! [`kernels::bind`], and it is SHARED: the column is the same column on all
//! three shader planes -- `shader_backends_agree` holds them identical --
//! so a per-driver reading of it would be three transcriptions of one
//! decision, which is the defect class this tree has now paid for five
//! times.
//!
//! What is left here is the half that is honestly per-driver: which handle a
//! FACT names. `keys::KvKeys` is an offset into this backend's paged pool
//! and a bind-group entry on wgpu; `keys::Rows` is a number on both but
//! comes off a different struct. That is [`Held::fact`], and the slot
//! accessors around it are this backend's [`Handles`].

use kernels::bind::{Answer, Holds};
use kernels::routine::Refusal;
use kernels::{Source, Ty};
use kernels_metal::routine::ArgValue;

use crate::lowering::executor::FireTable;
use crate::lowering::hold::{Facts, Handles};

/// Bind one launch's arguments from the row the signature derived.
///
/// A thin wrapper over [`kernels::bind::bind`]: it pairs this backend's
/// [`Handles`] with the launch's [`Facts`] so the shared reader can ask both
/// through one trait.
///
/// # Errors
///
/// [`Refusal::Unstated`] when an argument has no source, or has one this
/// backend cannot answer. Otherwise whatever the statement's own absences
/// produce: [`Refusal::Absent`] for a slot or scalar the trace does not
/// carry.
pub fn bind(
    args: &[Ty],
    sources: &[Option<Source>],
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    kernels::bind::bind::<ArgValue, _>(args, sources, &mut Held { o, f })
}

/// ONE value, for a body that ASKS rather than a column that declares.
///
/// The same resolver, entered at one argument instead of a list. Nothing new
/// answers — what changed is only where the question is asked from.
///
/// # Errors
///
/// [`Refusal::Unstated`] for a fact this backend does not answer, and whatever
/// the fact's own absence means otherwise.
pub fn one(
    ty: Ty,
    source: Source,
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<ArgValue, Refusal> {
    kernels::bind::one::<ArgValue, _>(ty, source, &mut Held { o, f })
}

/// This backend's answers, for the shared reader.
///
/// It borrows rather than owns the [`Handles`] because binding MUTATES them
/// -- every `input` numbers a handle -- and the caller keeps them afterwards
/// to build the encoder's binding list.
struct Held<'a, 'h> {
    o: &'h mut Handles<'a>,
    f: Facts,
}

impl Holds for Held<'_, '_> {
    fn input(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.input(n)
    }

    fn output(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.output(n)
    }

    fn output_read(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.output_read(n)
    }

    fn weight(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.weight(n)
    }

    // THE RECTANGLE, WHICH THE MARK NOW CARRIES. `shaped` asks for both
    // halves of an operand -- the handle and its row width -- and the default
    // here answers `Unstated`, which `bind` reads as a width of ZERO. Every
    // body that takes an `In<Tensor<_>>` and reads `x.width` then refuses
    // `Empty`, which is what left this backend unable to dispatch a rotation
    // or a strided activation.
    fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.in_width(n)
    }

    fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.out_width(n)
    }


    fn param(&self, n: usize) -> Result<i32, Refusal> {
        self.o.param(n)
    }

    fn param_f32(&self, n: usize) -> Result<f32, Refusal> {
        self.o.param_f32(n)
    }

    fn null(&mut self) -> u32 {
        self.o.state(None)
    }

    fn fact(&mut self, key: &'static str) -> Option<Result<Answer, Refusal>> {
        Some(named(key, self.o, self.f))
    }
}

/// A fact, by the key the signature names.
///
/// The tables and the pool come back as handles; the geometry comes back as
/// numbers off [`Facts`]. Both are this backend's to answer, which is what
/// `Provenance::Env` said all along -- this says WHICH.
///
/// It is total rather than optional at the outer layer: a key with no case
/// here is a `Refusal::Unstated` naming the fact, which reads better at a
/// fire than "the backend does not answer" would.
fn named(key: &'static str, o: &mut Handles<'_>, f: Facts) -> Result<Answer, Refusal> {
    use kernels::keys::{self, Fact};

    if let Some(which) = table_of(key) {
        return Ok(Answer::Handle(o.table(which)));
    }
    if key == keys::KvKeys::KEY {
        return Ok(Answer::Handle(o.kv(f.layer, false)));
    }
    if key == keys::KvValues::KEY {
        return Ok(Answer::Handle(o.kv(f.layer, true)));
    }
    for slab in SLABS {
        if key == slab {
            return Ok(Answer::Handle(o.slab(f.layer, slab)?));
        }
    }
    // THE MASK'S PITCH, which this driver answers ZERO and means it.
    //
    // Not a gap. `bind::tables` stages the enable flag as one zero word per
    // token and no mask beside it -- `Frame` has no field for a user mask --
    // so every row of every fire this driver builds is unmasked, and the
    // pitch of a mask with no rows is zero. The wgpu and vulkan planes stage
    // a real table and answer its real pitch; all three are asked the same
    // question and each answers its own fire, which is what unifies them.
    //
    // While this was `ParamOrLit<3, 0>` the same zero arrived by a different
    // road: the statement never carried scalar 3, so the literal won. The
    // number is unchanged and the sentence is now true.
    if key == keys::AttentionMaskStride::KEY {
        return Ok(Answer::Number(0));
    }
    // THE POOL'S NUMBERS reach a kernel as scalars rather than addresses,
    // and their absence is a refusal rather than a zero: a paged read with
    // no page size walks the wrong stride and answers fluently.
    //
    // `Wide` rather than `Number` because a stride is a byte count, and the
    // signature says whether it fits: `Usize` takes it whole and `U32`
    // refuses a deployment large enough to overflow, which is the honest
    // answer and was a silent truncation while the arms cast it.
    for (k, which, what) in POOLED {
        if key == k {
            let n = o.pooled(which).ok_or(Refusal::Unstated { what })?;
            return Ok(Answer::Wide(u64::from(n)));
        }
    }
    let n = geometry(key, f).ok_or(Refusal::Unstated {
        what: "a fact this backend does not answer",
    })??;
    Ok(Answer::Number(n.cast_signed()))
}

/// The slabs a key may name, which the driver holds per layer.
const SLABS: [&str; 3] = ["conv_state", "new_conv_state", "recurrent_state"];

/// The pool's numbers, which reach a kernel as scalars rather than addresses.
///
/// Their absence is a refusal rather than a zero: a paged read with no page
/// size walks the wrong stride and answers fluently.
const POOLED: [(&str, FireTable, &str); 3] = [
    (
        <kernels::keys::KvHeadStride as kernels::keys::Fact>::KEY,
        FireTable::KvHeadStride,
        "the KV head stride: the pool has none",
    ),
    (
        <kernels::keys::KvSeqStride as kernels::keys::Fact>::KEY,
        FireTable::KvSeqStride,
        "the KV sequence stride: the pool has none",
    ),
    (
        <kernels::keys::KvPageSize as kernels::keys::Fact>::KEY,
        FireTable::KvPageSize,
        "the KV page size: the pool has none",
    ),
];

/// The fire's GEOMETRY, by key. `None` for a key that is not one of the
/// fire's numbers, which is a refusal at the caller rather than here.
fn geometry(key: &str, f: Facts) -> Option<Result<u32, Refusal>> {
    use kernels::keys::{self, Fact};

    Some(Ok(match key {
        _ if key == keys::Rows::KEY => f.rows,
        _ if key == keys::RequestCount::KEY => f.requests,
        _ if key == keys::Width::KEY => f.width,
        _ if key == keys::InWidth::KEY => f.in_width,
        _ if key == keys::NumQHeads::KEY => f.q_heads(),
        _ if key == keys::NumKvHeads::KEY => f.kv_heads(),
        _ if key == keys::HeadDim::KEY => f.head_dim(),
        _ if key == keys::VHeads::KEY => f.v_heads(),
        _ if key == keys::VDim::KEY => f.v_dim(),
        _ if key == keys::NumExperts::KEY => f.n_experts(),
        _ if key == keys::ExpertsPerToken::KEY => f.experts_per_token(),
        _ if key == keys::QuantGroup::KEY => f.group(),
        _ if key == keys::QuantBits::KEY => f.bits(),
        // THE ROTARY WIDTH, which is the fallback half of every rope
        // routine's `ParamOr<3, ..>` and was missing here until a sweep of
        // the column asked for it by name. Every deployment in the suite
        // states its own, so the chain's second half never ran and nothing
        // went red -- see
        // `routine::tests::every_source_in_the_column_is_one_the_binder_answers`.
        _ if key == keys::RotaryWidth::KEY => f.rotary_dims(),
        // The tile is a pair behind an `Option`, and its absence is a real
        // refusal: a quantised matmul with no tile has no grid.
        _ if key == keys::TileM::KEY || key == keys::TileN::KEY => {
            let Some(tile) = f.tile else {
                return Some(Err(Refusal::Unstated {
                    what: "a tile: this device states none",
                }));
            };
            if key == keys::TileM::KEY { tile.0 } else { tile.1 }
        }
        _ => return None,
    }))
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
        _ if key == keys::RecurrentSlots::KEY => FireTable::RecurrentSlots,
        _ => return None,
    })
}
