//! Firing a row by name: the switch every generated dispatch arm goes through.
//!
//! **This is the whole point of the crate.** A kernel launch used to go: a
//! generated `match` arm, a generated `extern "C" pie_k_*`, a host launcher in
//! a `.cu` holding the `<<<>>>`, and an archive built by CMake to hold all
//! three. A row that comes HERE instead needs none of them — the template is
//! text in the binary, [`crate::runtime::nvrtc`] compiles it,
//! [`crate::runtime::cache`] keeps the module, and `cuLaunchKernel` takes a
//! `void**`. That is what lets a launcher be deleted: its only consumer was
//! the shim entry, and the shim entry's only consumer was the arm that now
//! calls [`fire`].
//!
//! # Why the caller passes a symbol and not a row
//!
//! Because it has a symbol. `model-compiler` writes one into a trace and the
//! dispatcher matches on it; a string is what a symbol IS at that boundary.
//! Handing [`fire`] a `&'static KernelSig` instead would mean the emitter
//! naming a table and an index — two facts about where a row LIVES — to save
//! a lookup the unit performs anyway while finding the module the fire needs.
//! The typed surface for callers that do have a row is [`crate::api`], which
//! is generated from the same table.
//!
//! # Why a failure is a refusal and not a fallback
//!
//! A row that reaches [`fire`] has no shim entry any more. If its unit will
//! not compile or its symbol will not resolve, there is nothing to fall back
//! *to* — a fallthrough would send the fire to a hand-written arm that does
//! not exist and be diagnosed as an unknown kernel, which is a lie about what
//! went wrong. So the diagnosis is reported with the unit and the symbol
//! named, and the refusal is returned.
//!
//! Both, and neither replaces the other: the log is for the OPERATOR, who is
//! looking at a process that will not serve a model and needs the unit name
//! and the compiler's words; the `Result` is for the CALLER, which needs to
//! know whether to try another dispatcher ([`Error::Unknown`]) or to stop
//! (everything else). The ahead-of-time `bind::jit::fire` had only the log,
//! and returned a `bool` that could not tell those apart.

use std::sync::OnceLock;

use crate::device::{Arm, DeviceKernel, Fact, Fault, Specialisation, Take};
use crate::runtime::{ArgValue, Args, Dims, Error, Stream, cache, launch};
use crate::unit;

/// How many operands a specialised variant may declare.
///
/// A fixed array rather than a `Vec`, because the reshape happens on the
/// launch path — once per kernel per layer per token — and an allocation
/// there to hold eight `Copy` words is the kind of cost this crate moved the
/// module cache to avoid. Twenty-four is past every row in the tree and is
/// checked rather than assumed: a variant declaring more is
/// [`Error::Specialise`], not a truncated argument list, because a truncated
/// `void**` is a launch that runs and reads its last operand from whatever
/// follows the array.
const SCRATCH: usize = 24;

/// What the predicate is allowed to see, out of what the fire is about to
/// bind.
///
/// One line, because the mapping itself is [`ArgValue::fact`] — it moved to
/// [`crate::runtime::args`], beside the variants it is a statement about, on
/// the day a `Bool` had to stop being [`Fact::Opaque`]. What stays here is the
/// SHAPE of the call: a fixed array, filled up to [`SCRATCH`], on the launch
/// path.
///
/// The enforcement point it used to be is unchanged and is documented on
/// [`ArgValue::fact`]: a pointer becomes an address and nothing downstream can
/// dereference it, and every kind a term may not read becomes
/// [`Fact::Opaque`] rather than a bit pattern that happens to divide by 8.
fn facts(values: &[ArgValue], into: &mut [Fact; SCRATCH]) -> usize {
    let n = values.len().min(SCRATCH);
    for (slot, value) in into.iter_mut().zip(&values[..n]) {
        *slot = value.fact();
    }
    n
}

/// Which arm a fire would take, without firing it.
///
/// The auditing surface, and the reason it is public: a specialised row makes
/// two kernels reachable through one symbol, and an operator looking at a
/// slow decode or a wrong number needs to be able to ask which one ran
/// without a profiler and without a launch. It is also how the negative
/// control in `tests/specialise.rs` shows the predicate answering `false` on
/// the cases the C++ answers `false` on.
///
/// `Ok(None)` is "the base row", which is every symbol that carries no
/// specialisation at all.
///
/// # Errors
///
/// [`Fault`] if a term names an operand this list does not have or cannot
/// read — which is drift between a `Select` and the row it is written
/// against, and is refused at fire time for the same reason.
pub fn selects(symbol: &str, values: &[ArgValue]) -> Result<Option<&'static Arm>, Fault> {
    let Some(spec) = crate::device::specialisation(symbol) else { return Ok(None) };
    let mut scratch = [Fact::Opaque; SCRATCH];
    let n = facts(values, &mut scratch);
    spec.choose(&scratch[..n])
}

/// Does any unit host `symbol`?
///
/// What a dispatcher asks before it emits an arm, and what a test asks to
/// check that a symbol routed to the JIT can actually be compiled by one. No
/// device is involved: this is a question about the TABLE, and it is
/// answerable on a machine with no GPU.
#[must_use]
pub fn hosts(symbol: &str) -> bool {
    unit::unit_of(symbol).is_some()
}

/// The row a symbol names, if any unit holds it.
///
/// For the generated typed façade in [`crate::api`], which needs the row's
/// signature to bind against, and for tests. Deliberately narrower than
/// [`crate::device::row`]: that one answers over every device table, this one
/// only over rows some unit will compile — and the difference between the two
/// answers is a row that can be described and not fired.
#[must_use]
pub fn row(symbol: &str) -> Option<&'static DeviceKernel> {
    unit::unit_of(symbol).and_then(|(_, unit)| unit.row(symbol))
}

/// Fire the row `symbol` names.
///
/// `dims` is the fire's rectangle, which is what every [`kernels::LaunchRule`]
/// is written over; the rule turns it into a grid and a block, and
/// [`crate::runtime::launch`] is the only place that arithmetic lives.
/// `values` are the row's operands in the row's order, checked against it by
/// [`Args::bind`] rather than trusted.
///
/// The first thing this does is look the symbol up, so a symbol no unit hosts
/// is refused before anything touches CUDA — which is what makes "not mine" a
/// cheap answer for a dispatcher that has other places to look.
///
/// # Errors
///
/// [`Error::Unknown`] if no unit hosts `symbol` — and that one alone means
/// *try somewhere else*. [`Error::NoDevice`], [`Error::Compile`] or
/// [`Error::Missing`] if the unit cannot be got onto the device;
/// [`Error::Geometry`] if the row's rule has no launch for these dims;
/// [`Error::Args`] if the values do not match the row; [`Error::Driver`] if
/// the launch itself is refused. Every one of them is also reported once per
/// symbol, at `error`, for whoever is reading the log rather than the return.
///
/// # Safety
///
/// `stream` must name a live CUDA stream for the duration of the launch, and
/// every [`ArgValue::Ptr`] in `values` must address device memory that is live
/// and large enough for the operand the row states — the launch is
/// asynchronous, so "for the duration" outlives this call and ends when the
/// stream is synchronised.
///
/// Nothing here can check either fact; the row's types are checked and its
/// pointers are not. This is the same obligation every `pie_k_*` call carried
/// when it handed `ctx.stream` and a set of arena offsets to a C++ launcher.
pub unsafe fn fire(
    symbol: &str,
    dims: Dims,
    values: &[ArgValue],
    stream: Stream<'_>,
) -> Result<(), Error> {
    let Some((index, unit)) = unit::unit_of(symbol) else {
        return Err(Error::Unknown { symbol: symbol.to_string() });
    };
    // The row is the unit's own, found by the symbol the arm matched on.
    // `unit_of` answered yes by finding it, so the `else` is unreachable by
    // construction and spelled anyway rather than unwrapped.
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        return Err(Error::Unknown { symbol: symbol.to_string() });
    };
    let outcome = (|| -> Result<(), Error> {
        let module = cache::module(index, unit)?;
        let geometry = launch::eval(sig.launch, dims)
            .map_err(|why| Error::Geometry { symbol: sig.symbol, why })?;
        let mut args = Args::bind(sig, values)?;
        // The specialisation is consulted AFTER the base row has been
        // resolved, its geometry evaluated and its values bound, and that
        // order is the invariant that keeps a row a contract: a fire refused
        // for a bad argument list is refused identically, with the same
        // error, naming the same symbol, whether or not the row carries an
        // arm. Specialisation may change which kernel runs. It may not change
        // which fires are legal.
        if let Some(spec) = crate::device::specialisation(symbol)
            && let Some(arm) = choose(spec, values)?
        {
            return fire_arm(module, arm, dims, values, stream);
        }
        // [`KernelModule::fire`] is safe and this function is not, and the
        // difference is exactly the obligation the doc above states: the
        // module checks that the entry is its own and that the geometry is
        // non-empty, and nothing anywhere checks that an `ArgValue::Ptr`
        // addresses memory this kernel may read or write. This signature is
        // where a caller says it does.
        module.fire(sig, geometry, &mut args, stream)
    })();
    if let Err(why) = &outcome {
        report(symbol, unit.name, &why.to_string());
    }
    outcome
}

/// The arm this fire takes, with a [`Fault`] turned into the crate's refusal.
///
/// Separate from [`selects`] only so that the launch path names
/// [`Error::Specialise`] and the auditing surface names the [`Fault`] itself:
/// a test asking which arm a list chooses wants the fault's own words, and a
/// dispatcher wants one error type.
fn choose(
    spec: &'static Specialisation,
    values: &[ArgValue],
) -> Result<Option<&'static Arm>, Error> {
    let mut scratch = [Fact::Opaque; SCRATCH];
    let n = facts(values, &mut scratch);
    spec.choose(&scratch[..n]).map_err(|why| Error::Specialise {
        symbol: spec.base,
        why: format!(
            "the selection predicate could not be evaluated against this row: {why} — \
             which means the terms and the operand list have drifted"
        ),
    })
}

/// Launch the arm the predicate chose.
///
/// The variant's values are the base's, moved by [`Take`] into the variant's
/// parameter order and bound against the VARIANT's own signature — so the
/// kinds are checked twice, once for the row the caller named and once for
/// the kernel that actually runs. That second check is not redundant: the two
/// templates have different parameter lists, and the whole hazard of a
/// specialisation is a reshape that puts a stride where a width belongs.
/// [`crate::device::Specialisation::agrees`] proves the reshape is well typed
/// with no GPU; this is what happens when it is not.
///
/// The geometry is the VARIANT's rule evaluated over the same dims. `agrees`
/// requires the two rules to be equal, so this is the base's launch by
/// construction — evaluated through the variant's row rather than reused, so
/// that the launch a kernel gets always comes from the row that named it.
fn fire_arm(
    module: &'static crate::runtime::KernelModule,
    arm: &'static Arm,
    dims: Dims,
    values: &[ArgValue],
    stream: Stream<'_>,
) -> Result<(), Error> {
    let sig = arm.row.sig;
    if arm.take.len() > SCRATCH {
        return Err(Error::Specialise {
            symbol: sig.symbol,
            why: format!(
                "the arm takes {} arguments and this launch path carries {SCRATCH}",
                arm.take.len()
            ),
        });
    }
    let mut reshaped = [ArgValue::Ptr(std::ptr::null_mut()); SCRATCH];
    for (slot, take) in arm.take.iter().enumerate() {
        reshaped[slot] = match take {
            Take::From(index) => match values.get(*index) {
                Some(value) => *value,
                None => {
                    return Err(Error::Specialise {
                        symbol: sig.symbol,
                        why: format!(
                            "the arm fills argument {slot} from operand {index}, and the fire \
                             bound {}",
                            values.len()
                        ),
                    });
                }
            },
            Take::Null => ArgValue::Ptr(std::ptr::null_mut()),
        };
    }
    let geometry =
        launch::eval(sig.launch, dims).map_err(|why| Error::Geometry { symbol: sig.symbol, why })?;
    let mut args = Args::bind(sig, &reshaped[..arm.take.len()])?;
    // SAFETY-adjacent, and the same obligation `fire` states: the pointers
    // are the caller's, unchanged -- a reshape moves values, it does not
    // invent them, and the one value it does invent is a null on an operand
    // the row declares nullable.
    module.fire(sig, geometry, &mut args, stream)
}

/// Say what went wrong, once per symbol.
///
/// Once, because a fire is per layer per token and a broken unit would
/// otherwise produce a line per launch — which is how a real diagnosis becomes
/// unreadable, and how the one line that named the actual compiler error ends
/// up ten thousand lines above the point anyone starts reading.
///
/// This does not replace returning the error. It is the operator's copy.
fn report(symbol: &str, unit: &str, why: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SAID: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let said = SAID.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut said) = said.lock()
        && said.insert(symbol.to_string())
    {
        tracing::error!(symbol, unit, why, "a device unit will not fire");
    }
}

#[cfg(test)]
mod tests {
    use super::{fire, hosts, row};
    use crate::runtime::{Dims, Error, Stream};
    use crate::unit::{UNITS, unit_of};

    /// A symbol no unit holds. Named for what it is, so that a day when some
    /// unit does hold it is a day this test is wrong on purpose.
    const NOBODY: &str = "norm::a_kernel_nobody_wrote";

    /// The rectangle the refusal tests fire over. Non-empty, so that nothing
    /// they assert can be explained by [`crate::runtime::Ungeometric::Empty`]
    /// — which now includes the head and expert axes, because the rules that
    /// read them refuse a zero rather than launching a grid of nothing.
    const DIMS: Dims = Dims {
        rows: 1,
        width: 1,
        in_width: 1,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 2,
        // The tenth axis, and the one whose zero is not a refusal: these
        // fires state no per-head width, which is `LaunchRule::RowsPerHead`'s
        // absent arm and every other rule's don't-care.
        stated_head_dim: 0,
        rotary_dims: 2,
        n_experts: 1,
        experts_per_token: 1,
        // The eleventh and twelfth. Both are non-zero for the same reason the
        // eight above are: `LaunchRule::PagedScores` and `AltUpStreams`
        // refuse a zero request count and a zero stream count, and a refusal
        // test that got `Ungeometric::Empty` from an unset fixture would pass
        // without touching the thing it names.
        requests: 1,
        altup_streams: 1,
    };

    /// [`hosts`] is [`unit_of`] with the answer thrown away, and a dispatcher
    /// emits arms on the strength of it — so the two agreeing is the property
    /// that keeps an emitted arm from calling a [`fire`] that refuses.
    #[test]
    fn hosts_agrees_with_the_unit_table() {
        for unit in UNITS {
            for entry in unit.rows {
                assert!(
                    hosts(entry.sig.symbol),
                    "{} is in a unit and not hosted",
                    entry.sig.symbol
                );
                assert!(unit_of(entry.sig.symbol).is_some());
            }
        }
        assert_eq!(hosts(NOBODY), unit_of(NOBODY).is_some());
    }

    /// The negative half, which is the one a dispatcher relies on: a symbol
    /// this crate does not host must be answerable as "not mine" without a
    /// device, a compile, or a guess.
    #[test]
    fn a_symbol_no_unit_holds_is_not_hosted() {
        assert!(!hosts(NOBODY));
        assert!(row(NOBODY).is_none());
    }

    /// [`row`] answers with the row that was asked for and not merely with a
    /// row — the failure it rules out is a lookup that finds the unit and
    /// returns its first entry, which would bind the wrong operands and fire
    /// something that runs.
    #[test]
    fn a_row_is_the_one_asked_for() {
        for unit in UNITS {
            for entry in unit.rows {
                let found = row(entry.sig.symbol).expect("a unit's own row");
                assert_eq!(found.sig.symbol, entry.sig.symbol);
            }
        }
    }

    /// An unknown symbol is refused before CUDA is touched.
    ///
    /// Which is why this test runs on a machine with no GPU: [`fire`] looks
    /// the symbol up first, and [`Error::Unknown`] is returned from the
    /// lookup rather than from a launch that failed. A restructuring that
    /// discovered the architecture first would still be correct and would
    /// make this test require a driver — that is the signal it is here for.
    #[test]
    fn an_unknown_symbol_is_refused_without_a_device() {
        // SAFETY: no launch happens -- the symbol is refused before any
        // pointer or stream is read, which is what this test asserts.
        let refusal = unsafe { fire(NOBODY, DIMS, &[], Stream::NULL) };
        assert_eq!(refusal, Err(Error::Unknown { symbol: NOBODY.to_string() }));
    }
}
