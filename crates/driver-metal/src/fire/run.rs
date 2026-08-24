//! Running a fire: what is left to do once the walk has walked.
//!
//! ```text
//! Baked::lane(class, masked)    the fact word picks a Program   baker
//! Fire::walk(&Encoder)          steps -> dispatches             baker
//! stage / compile / encode      dispatches -> a command buffer  here
//! ```
//!
//! # What left this module
//!
//! `rows_of`, `lower` and `plan`. The legacy walk flattened a TEXT once per
//! fire — rows from the region table, rectangles from the text, then a grid
//! per launch read off a `kernel!` row's launch rule — and all three are
//! gone. A `Program` is bound at LOAD, a fire picks one by fact word, and a
//! grid is computed inside the claim body that fires it. What is left is the
//! three device things a list of dispatches still needs: the scalars staged,
//! the pipelines compiled, and the commands encoded.
//!
//! Nothing here decides anything.

use crate::baker::dispatch::{Dispatch, table_width};
use crate::baker::walk::Blit;
use crate::bind::encode::{Params, Pipelines, encode};
use crate::device::{ArgumentTable, Context, Stepper};
use crate::error::{Error, Result};
use crate::fire::scratch::Lease;
use crate::layout::region::Region as _;
use crate::program::Compiler;

/// Everything a driver keeps ACROSS fires.
///
/// Six things, each wrong to rebuild per fire: `stepper` (the timeline and
/// allocator ring — fresh means no value to compare against, no pipelining),
/// `scratch` (the fire's regions — fresh leaks into the residency set
/// permanently and moves an address that should stay stable), `pipelines`
/// (the compile cache — rebuilding recompiles every shader), `regions` and
/// `recordings` (the replay path), and `context`/`compiler` (the device and
/// its shader compiler).
///
/// Grouped rather than passed as separate parameters because a caller that
/// has one has all of them.
#[derive(Debug)]
pub struct Machine<'c, 's> {
    /// The device.
    pub context: &'c Context,
    /// Its shader compiler.
    pub compiler: &'c Compiler,
    /// The compiled-pipeline cache.
    pub pipelines: &'c mut Pipelines,
    /// The command timeline and allocator ring.
    pub stepper: &'c mut Stepper<'s>,
    /// The reusable fire regions.
    pub scratch: &'c crate::fire::Scratch,
    /// Which buffer each address belongs to, for recording.
    pub regions: &'c mut crate::device::Regions,
    /// Fires already recorded, by what they are valid for.
    ///
    /// `None` means **do not try**: a recording binds buffers, so it can only
    /// be made once every region a fire's operands point into — the weights
    /// and the KV pool, owned by the caller — is registered.
    pub recordings: Option<&'c mut crate::fire::Recordings>,
}

/// One committed fire: the timeline value to wait on, and what must outlive
/// it.
pub struct InFlight {
    /// The value the stepper will signal.
    pub value: u64,
    /// The activation arena, held because the read-out is a span of it.
    pub arena: Lease,
    _table: ArgumentTable,
    _params: Params,
}

/// Stage the operand copies an `InOut` point forced.
///
/// A HOST MEMCPY, AND ON THIS PLANE THAT IS THE HONEST FORM. Cuda issues a
/// `cudaMemcpyAsync` on the fire's stream because its arena is device-only;
/// every allocation this driver makes is `StorageModeShared`, which on Apple
/// silicon is memory the host and the GPU both address, so the copy is a
/// `memcpy` and needs no encoder, no queue and no barrier.
///
/// **The ordering that makes it sound is the LEASE, not luck.** Every `from`
/// and every `to` is a span of the arena this fire leased from
/// `Shell::scratch`; a lease is held for as long as the fire is in flight
/// ([`InFlight::arena`]); and nothing has been committed yet. So no GPU
/// command anywhere can be reading these bytes while this loop runs. A fire
/// that shared its arena with one still in flight would break that, which is
/// exactly what the lease exists to prevent.
///
/// # Errors
///
/// A span that leaves the arena, which is drift between the walk's offsets
/// and the arena it was walked against.
fn stage_blits(arena: &Lease, blits: &[Blit]) -> Result<()> {
    let base = arena.gpu_address();
    for b in blits {
        let (Some(dst), Some(src)) = (
            b.to.address.checked_sub(base),
            b.from.address.checked_sub(base),
        ) else {
            return Err(Error::Unserved {
                what: "fire",
                message: format!(
                    "op {} stages an in-place operand from outside this fire's arena",
                    b.op
                ),
            });
        };
        // SAFETY: both spans are checked against the arena's own length by
        // `Region::copy` before a byte moves, and nothing is encoded against
        // the arena yet — see this function's own note on the lease.
        unsafe { arena.copy(dst, arena.region(), src, b.bytes) }?;
    }
    Ok(())
}

/// Stage, compile, encode and COMMIT one fire, without waiting for it.
///
/// `arena` is the caller's lease and must be THE ONE THE WALK WAS WALKED
/// AGAINST: every `Dispatch`'s operands are addresses inside it, computed
/// before this is called, so allocating one here would plan against one arena
/// and bind another. That is the one thing this signature is shaped to make
/// impossible, and the legacy `submit` — which allocated and then planned —
/// carried a paragraph about the same hazard.
///
/// # Errors
///
/// A blit that leaves the arena, a shader that will not compile, or any
/// device failure on the way.
pub fn submit(
    machine: &mut Machine<'_, '_>,
    arena: Lease,
    dispatches: &[Dispatch],
    blits: &[Blit],
) -> Result<InFlight> {
    let Machine {
        context,
        compiler,
        pipelines,
        stepper,
        scratch,
        regions,
        recordings,
    } = machine;
    stage_blits(&arena, blits)?;
    let params = Params::stage_in(context, scratch, dispatches)?;
    // What this fire leased, so a recording can turn its operand addresses
    // back into buffers. Pooled, so re-registering is the same span and the
    // registry does not grow.
    regions.add(&arena);
    regions.add(params.region());
    let table = ArgumentTable::new(context, table_width(dispatches))?;
    pipelines.ensure(context, compiler, dispatches)?;
    // REPLAY if this exact fire has been recorded (roughly two orders of
    // magnitude cheaper than re-encoding every dispatch), else ENCODE.
    //
    // The fingerprint is the validity condition, checked rather than assumed:
    // a recording bakes each operand's buffer and offset, its grid and its
    // pipeline, so replaying one against a fire that differs in any of those
    // would silently run the wrong program.
    //
    // A RECORDING CANNOT CARRY A BLIT, and it does not have to: the copies
    // are staged above, on the host, every time — before a replay exactly as
    // before an encode.
    //
    // `Error::Unrecordable` (an operand in no registered allocation) falls
    // back to encoding on purpose; anything else `record` can fail at is a
    // bug and propagates.
    let recorded = match recordings.as_mut() {
        None => None,
        Some(recordings) => {
            match recordings.get_or_record(context, pipelines, &params, regions, dispatches) {
                Ok(r) => Some((objc2::rc::Retained::from(r.buffer()), r.commands())),
                Err(Error::Unrecordable { .. }) => None,
                Err(other) => return Err(other),
            }
        }
    };
    let value = match recorded {
        Some((icb, commands)) => stepper.submit(|encoder| {
            encoder.execute_commands(&icb, 0..commands)?;
            // The encode path closes its fire with one; a replay must too, or
            // whatever reads the arena next races the last recorded command.
            encoder.barrier(crate::device::Visibility::Device);
            Ok(())
        })?,
        None => {
            stepper.submit(|encoder| encode(encoder, &table, pipelines, &params, dispatches))?
        }
    };
    Ok(InFlight {
        value,
        arena,
        _table: table,
        _params: params,
    })
}
