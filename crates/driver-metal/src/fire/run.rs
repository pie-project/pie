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
/// The `InOut` copies, as DISPATCHES, each in front of the statement that
/// asked for it.
///
/// # What stood here, and why it was wrong
///
/// `stage_blits`: every operand copied on the HOST before a single dispatch of
/// the fire had run, under a SAFETY note that said so — "nothing is encoded
/// against the arena yet". That is correct only if an operand's bytes are
/// already final when the fire starts, and they are not.
/// `norm.residual_add` at op14 of a qwen3.5 decode takes value 1, **the
/// output of `layout.embed`, which is dispatch 0 of this same fire**. The copy
/// happened first, so the residual stream was added to the zeros the arena was
/// cleared to, from layer 0 onward.
///
/// It was silent: nothing refused, a whole 24-layer tower fired, and the
/// answer was a real forward of something else — 93126 at 10.9375 where cuda
/// banked 198 at 12.3125. A bisect against `driver-wgpu` on the identical
/// program found it at the first `InOut` statement the tower states, with
/// everything before it matching to within a bf16 ulp.
///
/// # Why a dispatch rather than a blit encoder
///
/// A `StepEncoder` is a COMPUTE encoder and Metal does not copy from one. The
/// alternatives were a second pass per copy — sixty-six submissions for a
/// qwen3.5 decode — or this: a copy that IS a dispatch, ordered by the same
/// hazard tracker as every other statement and recorded into the same indirect
/// command buffer. The paragraph in [`submit`] that said "a recording cannot
/// carry a blit, and it does not have to" is answered by there being no blits
/// left, only dispatches.
///
/// # Errors
///
/// A copy filed against a statement that planned no dispatch, or one whose
/// span is not a whole number of `ushort` — `layout/blit.metal` says why the
/// element is two bytes.
fn with_blits(dispatches: &[Dispatch], blits: &[Blit]) -> Result<Vec<Dispatch>> {
    if blits.is_empty() {
        return Ok(dispatches.to_vec());
    }
    let unserved = |message: String| Error::Unserved {
        what: "fire",
        message,
    };
    // Filed against the FIRST dispatch of the statement that asked. A body may
    // state more than one launch — `rope.full` is two — and the operand's
    // bytes have to be in place before the first of them writes through the
    // handle, not before the last.
    let mut ahead: std::collections::BTreeMap<usize, Vec<Dispatch>> = Default::default();
    for b in blits {
        let at = dispatches
            .iter()
            .position(|d| d.op == b.op)
            .ok_or_else(|| {
                unserved(format!(
                    "op {} stages an in-place operand and planned no dispatch",
                    b.op
                ))
            })?;
        if !b.bytes.is_multiple_of(2) {
            return Err(unserved(format!(
                "op {} stages {} bytes, which is not a whole number of the \
                 two-byte element `blit_bfloat16` moves",
                b.op, b.bytes
            )));
        }
        let elements = u32::try_from(b.bytes / 2).map_err(|_| {
            unserved(format!(
                "op {} stages {} bytes, past a u32 of elements",
                b.op, b.bytes
            ))
        })?;
        ahead.entry(at).or_default().push(Dispatch {
            symbol: "blit_bfloat16",
            file: "layout/blit.metal",
            stamp: "",
            grid: [elements, 1, 1],
            threadgroup: [elements.clamp(1, 256), 1, 1],
            args: vec![
                crate::baker::BoundRegion {
                    slice: b.from,
                    width: 0,
                },
                crate::baker::BoundRegion {
                    slice: b.to,
                    width: 0,
                },
                crate::baker::NOTHING,
            ],
            // READ THE OPERAND, WRITE THE RESULT, and stated rather than
            // conservative: this is the one dispatch in a fire whose direction
            // is known exactly, and an honest pair is what lets the tracker put
            // a barrier between the statement that produced the operand and
            // this copy.
            touches: crate::baker::dispatch::Touches {
                reads: vec![b.from],
                writes: vec![b.to],
            },
            param_slots: vec![crate::baker::dispatch::ParamSlot {
                slot: 2,
                at: 0,
                bytes: 4,
                value: 0,
            }],
            params: vec![elements],
            layers: dispatches[at].layers.clone(),
            op: b.op,
        });
    }
    let mut out = Vec::with_capacity(dispatches.len() + blits.len());
    for (i, d) in dispatches.iter().enumerate() {
        if let Some(copies) = ahead.remove(&i) {
            out.extend(copies);
        }
        out.push(d.clone());
    }
    Ok(out)
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
    // THE COPIES ARE DISPATCHES NOW — see `with_blits`, which is where the
    // host-side staging this line used to do is written up. Everything below
    // sees one list, so the argument table, the fingerprint, the recording and
    // the barriers all take them as the statements they are.
    let merged = with_blits(dispatches, blits)?;
    let dispatches = merged.as_slice();
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
                // THE REFUSAL GETS A READER, behind a switch. `Recordings::
                // refusals` was written for "whoever has to say so" and had
                // nobody: serving swallows this on purpose — an unrecordable
                // fire is encoded instead, which is slower and right — so a
                // deployment paying the encode every step had no way to learn
                // it. The file's own words for what that costs are "a 374x
                // slowdown deserves better than a count".
                Err(err @ Error::Unrecordable { .. }) => {
                    #[allow(
                        clippy::print_stderr,
                        reason = "a probe that says why a fire re-encodes is the job"
                    )]
                    if std::env::var_os("PIE_METAL_TRACE_RECORD").is_some() {
                        eprintln!(
                            "PIE_RECORD refused ({} dispatches, {} recorded so far): {err}",
                            dispatches.len(),
                            recordings.recorded(),
                        );
                    }
                    None
                }
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
