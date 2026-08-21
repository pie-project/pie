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

use kernels::Ty;
use kernels::bind::Holds;
use kernels::bind::{Answer, Holds};
use kernels::routine::Refusal;
use kernels_vulkan::routine::ArgValue;

use crate::binding::FireTable;
use crate::hold::{Facts, Handles};

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
    views: &mut crate::views::Views<'_>,
) -> Result<Vec<ArgValue>, Refusal> {
    // Argument by argument rather than the shared list reader, because ONE
    // carrier is this plane's to answer before the reader sees it: a
    // `Ty::Raised` operand is a HOST view the driver builds
    // (`crate::views`), and the shared binder has no door for a value that
    // is neither a handle nor a scalar. Everything else goes through
    // `kernels::bind::one` exactly as the list reader would have sent it —
    // same order, same slot numbering, same refusals.
    let mut out = Vec::with_capacity(args.len());
    for (at, ty) in args.iter().enumerate() {
        let source = sources
            .get(at)
            .copied()
            .flatten()
            .ok_or(Refusal::Unstated {
                what: "an argument whose signature does not say where it comes from",
            })?;
        if matches!(ty, Ty::Raised) {
            out.push(views.raise(source, o, f)?);
            continue;
        }
        let mut held = Held { o, f };
        out.push(kernels::bind::one::<ArgValue, _>(*ty, source, &mut held)?);
    }
    Ok(out)
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

    fn param(&self, n: usize) -> Result<i32, Refusal> {
        self.o.param(n)
    }

    fn param_f32(&self, n: usize) -> Result<f32, Refusal> {
        self.o.param_f32(n)
    }

    fn null(&mut self) -> u32 {
        self.o.unbound()
    }

    fn rows(&mut self) -> i32 {
        // The launch rectangle's — what `keys::Rows` used to answer.
        self.f.rows.cast_signed()
    }
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
    ///
    /// WHERE THE FACTS WENT. Before the marks, a signature named a fact by
    /// TYPE -- `Held<keys::KvPageSize>` -- and `#[routine]` wrote the name
    /// into `sources`, so the column below was the whole census. It is not
    /// any more: a fact is asked for in the BODY now, `ctx.ask::<i32,
    /// keys::KvPageSize>()`, and a body is code rather than a table. What is
    /// left in the column is the handful of sources a MARK still derives --
    /// the two positional weights `Const<Tensor<E>>` spells and the recurrent
    /// slot table -- and this gate holds those. The body half is measured on
    /// a device by `tests/arena.rs`, which runs every routine's body against
    /// a real fire and names each refusal.
    #[test]
    fn every_fact_this_backend_s_kernels_name_is_one_it_answers() {
        let f = facts_for_test();
        let mut unanswered: Vec<&'static str> = Vec::new();
        let mut named: Vec<&'static str> = Vec::new();
        let mut asked = 0usize;
        let mut whole = 0usize;
        let mut routines = 0usize;

        for routine in kernels_vulkan::routines() {
            routines += 1;
            let mut bad = 0usize;
            for source in routine.sources {
                for key in keys_of(source.as_ref()) {
                    asked += 1;
                    if !named.contains(&key) {
                        named.push(key);
                    }
                    if whence(key, f).is_none() {
                        bad += 1;
                        if !unanswered.contains(&key) {
                            unanswered.push(key);
                        }
                    }
                }
            }
            if bad == 0 {
                whole += 1;
            }
        }

        println!("routines whose column facts are all answered: {whole} of {routines}");
        println!("distinct keys named by the column: {named:?}");
        assert_eq!(
            named,
            ["weight", "weight2"],
            "the column names a key it did not before -- a mark grew a source"
        );
        println!("of those, unanswered: {unanswered:?}");
        // The denominator. A column read as empty satisfies every line below,
        // and a `sources` accessor that changed shape is exactly how that
        // happens. Ninety-odd, not the three-hundred-odd of the era when a
        // signature named its facts -- see the note above.
        assert!(
            asked > 50,
            "{asked} sources read off this backend's column -- the sweep found \
             almost nothing, so the emptiness below is the reader's and not \
             the column's"
        );
        // WHICH KEYS this backend cannot answer from the column, named rather
        // than counted, and by KEY rather than by routine: two names account
        // for all ninety-odd of them, and listing the routine-and-slot pairs
        // would make this a diff of the kernel crate.
        //
        // `weight` and `weight2` are rule E6's NAMED weights, which
        // `Const<Tensor<E>>` derives as `Or(Named("weight"), Slot(Weight,
        // 0))`. They are resolved from the statement's `LaunchSpec::weight`
        // before a `Cx` exists -- `bind/mod.rs`'s `resolve_arg_windowed` --
        // so `whence`, which answers a fact out of the FIRE, is the wrong
        // half to ask, and its `None` is the right answer rather than a hole.
        // The wrong answer would be a handle to something else.
        //
        // `recurrent_slots` used to be here, on the five `gdn_*` routines,
        // because this driver does not serve recurrent state at all --
        // `frames.rs` refuses a plan carrying `rs_slot_ids`. It left the
        // column with the marks, not the crate: those bodies ask for it now,
        // and `named` still refuses it by the same arm.
        assert_eq!(
            unanswered,
            ["weight", "weight2"],
            "a different set of keys is named by this backend's column but \
             not answered by `whence`. If one LEFT, a weight stopped being \
             named; if one ARRIVED, a mark started deriving a source \
             `whence` has never heard of.",
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
