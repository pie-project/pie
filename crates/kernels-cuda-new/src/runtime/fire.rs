use std::sync::OnceLock;

use crate::device::{Arm, DeviceKernel, Fact, Fault, Specialisation, Take};
use crate::runtime::{ArgValue, Args, Dims, Error, Stream, cache, launch};
use crate::unit;

/// How many operands a specialised variant may declare.
const SCRATCH: usize = 24;

/// What the predicate is allowed to see, out of what the fire is about to
fn facts(values: &[ArgValue], into: &mut [Fact; SCRATCH]) -> usize {
    let n = values.len().min(SCRATCH);
    for (slot, value) in into.iter_mut().zip(&values[..n]) {
        *slot = value.fact();
    }
    n
}

/// Which arm a fire would take, without firing it.
pub fn selects(symbol: &str, values: &[ArgValue]) -> Result<Option<&'static Arm>, Fault> {
    let Some(spec) = crate::device::specialisation(symbol) else { return Ok(None) };
    let mut scratch = [Fact::Opaque; SCRATCH];
    let n = facts(values, &mut scratch);
    spec.choose(&scratch[..n])
}

/// Does any unit host `symbol`?
#[must_use]
pub fn hosts(symbol: &str) -> bool {
    unit::unit_of(symbol).is_some()
}

/// The row a symbol names, if any unit holds it.
#[must_use]
pub fn row(symbol: &str) -> Option<&'static DeviceKernel> {
    unit::unit_of(symbol).and_then(|(_, unit)| unit.row(symbol))
}

/// Fire the row `symbol` names.
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
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        return Err(Error::Unknown { symbol: symbol.to_string() });
    };
    let outcome = (|| -> Result<(), Error> {
        let module = cache::module(index, unit)?;
        let geometry = launch::eval(sig.launch, dims)
            .map_err(|why| Error::Geometry { symbol: sig.symbol, why })?;
        let mut args = Args::bind(sig, values)?;
        if let Some(spec) = crate::device::specialisation(symbol)
            && let Some(arm) = choose(spec, values)?
        {
            return fire_arm(module, arm, dims, values, stream);
        }
        module.fire(sig, geometry, &mut args, stream)
    })();
    if let Err(why) = &outcome {
        report(symbol, unit.name, &why.to_string());
    }
    outcome
}

/// The arm this fire takes, with a [`Fault`] turned into the crate's refusal.
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
    module.fire(sig, geometry, &mut args, stream)
}

/// Say what went wrong, once per symbol.
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
    const NOBODY: &str = "norm::a_kernel_nobody_wrote";

    /// The rectangle the refusal tests fire over. Non-empty, so that nothing
    const DIMS: Dims = Dims {
        rows: 1,
        width: 1,
        in_width: 1,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 2,
        stated_head_dim: 0,
        rotary_dims: 2,
        n_experts: 1,
        experts_per_token: 1,
        requests: 1,
        altup_streams: 1,
    };

    /// [`hosts`] is [`unit_of`] with the answer thrown away, and a dispatcher
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
    #[test]
    fn a_symbol_no_unit_holds_is_not_hosted() {
        assert!(!hosts(NOBODY));
        assert!(row(NOBODY).is_none());
    }

    /// [`row`] answers with the row that was asked for and not merely with a
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
    #[test]
    fn an_unknown_symbol_is_refused_without_a_device() {
        // SAFETY: no launch happens -- the symbol is refused before any
        let refusal = unsafe { fire(NOBODY, DIMS, &[], Stream::NULL) };
        assert_eq!(refusal, Err(Error::Unknown { symbol: NOBODY.to_string() }));
    }
}
