//! A load's fire and context, fitted to its device before anything is
//! reserved on it.
//!
//! A load that states no fire (`max_tokens` 0) takes the largest rung its
//! device holds: from the most rows the device dispatches on an axis (a
//! kernel puts a fire's rows on one), halving until a fire's activation
//! span (the carve and the kernel scratch rooms, [`crate::arena`]) takes no
//! more than the fire's share of what the device has left past the weights
//! ([`Tactics::working_share`](crate::backend::Tactics::working_share), the
//! one place that policy is stated) and the pool the load lays beside it
//! holds ([`pool`], the load's own sizing). The rest is the state's, as on
//! the cuda shell, whose halving and whose share this is
//! ([`engine::fit::tokens_within`], `engine.working_set_percent`). A
//! recurrent model's fire replays its rows in a scratch outside the pool
//! ([`rs_scratch`]), which is the fire's too: it counts against the share
//! with the span, and the pool is laid in what it leaves ([`takes`]). The
//! halving stops at the smallest rung that carries whole the widest run its
//! own ladders admit ([`floor`]): one token for a model of text alone. A fire
//! carries the lanes the load asks, no more than its rung has tokens: the
//! span is priced with them.
//!
//! A context the model declares, where the operator states none, is
//! shortened to the longest at which one sequence's working set fits the
//! same share ([`engine::fit::pages_within`]); a stated one is taken as it
//! is, and refused by name where it does not fit.
//!
//! This is the load's alone: nothing shrinks a fire or a context once the
//! load has them. The load says once what it fitted and what held each
//! figure there ([`Stop`]).

use engine::load::{Budgets as LoadBudgets, Residency};
use poem_compiler::{Budgets, DeviceProfile, compile_axes};
use poem_ir::Trace;

use crate::arena::Span;
use crate::backend::{Context as _, Tactics as _};
use crate::error::{Fault, Result};
use crate::serve::Serve;
use crate::store::Rows;
use crate::store::kv::{self, Paging};

/// The chunks one working set takes beside the null chunk at `asked`'s
/// context: its pages, its record and one written copy-on-write, its
/// rs-list pages and its window lists'.
#[must_use]
pub fn least_chunks(rows: &Rows, asked: Paging) -> u32 {
    let chunk = rows.chunk_bytes;
    let (rs_bytes, rs_tokens) = rows.rs.as_ref().map_or((0, 0), |rs| (rs.bytes, rs.tokens));
    crate::store::layout::Layout::least_chunks(
        chunk,
        [rows.list_bytes, rows.fixed_bytes],
        rows.pieces,
        asked.pages_per_slot,
    ) + chunk.checked_div(rs_bytes).map_or(0, |per| {
        asked
            .context()
            .div_ceil(rs_tokens.max(1))
            .div_ceil(u32::try_from(per).unwrap_or(1).max(1))
    }) + rows
        .windows
        .iter()
        .map(|w| {
            let per = u32::try_from(chunk / w.bytes.max(1)).unwrap_or(1).max(1);
            asked.context().div_ceil(w.tokens.max(1)).div_ceil(per)
        })
        .sum::<u32>()
}

/// The chunk pool a load lays: its chunks, how its segments cut them, and
/// what it was sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool {
    /// The bytes the pool was sized within.
    pub room: u64,
    /// Its chunks, the top `act` the activation span's.
    pub total: u32,
    /// The chunks one segment holds.
    pub segment_chunks: u32,
    /// The chunks the kernels' bf16-element index addresses: state lies
    /// within them.
    pub words: u32,
}

/// The pool a device with `left` bytes holds, chunks of `chunk`: the bytes
/// `kv_budget` states for state beside the span's `act` chunks, or all of
/// `left`, in whole chunks; at most four segments, none past `binding`
/// (what one buffer spans), the span whole in the last, and state within
/// the kernels' index. Refused by name where one working set
/// (`least_chunks`) does not stand beside the span.
pub fn pool(
    left: u64,
    kv_budget: Option<u64>,
    chunk: u64,
    act: u32,
    least_chunks: u32,
    binding: u64,
) -> Result<Pool> {
    let room = kv_budget.map_or(left, |budget| {
        budget.saturating_add(u64::from(act) * chunk).min(left)
    });
    let total = engine::chunk::fit(room, chunk, act, least_chunks)
        .map_err(|short| Fault::Residency(short.to_string()))?;
    let binds = u32::try_from((binding / chunk).saturating_mul(kernels_slang::SEGMENTS as u64))
        .unwrap_or(u32::MAX);
    // Whole segments of a quarter of the pool, all four within the
    // bf16-element index the kernels address the pool by.
    let words = u32::try_from(2 * u64::from(u32::MAX) / chunk).unwrap_or(u32::MAX)
        / kernels_slang::SEGMENTS as u32
        * kernels_slang::SEGMENTS as u32;
    // State lies within that index; the span past it is the fire's
    // alone (bound by offset, never indexed), at every rung.
    let total = total.min(binds).min(words.saturating_add(act));
    let span_short = || {
        Fault::Residency(format!(
            "one working set takes {least_chunks} chunks of {chunk} bytes beside a fire's \
             {act}, and this device binds four segments of {} MiB",
            binding >> 20
        ))
    };
    let segment_chunks =
        crate::store::pool::segment_chunks(chunk, total, binding, act).ok_or_else(span_short)?;
    if total < least_chunks + act {
        return Err(span_short());
    }
    Ok(Pool {
        room,
        total,
        segment_chunks,
        words,
    })
}

/// The replay scratch a recurrent model's fire of `tokens` over `lanes`
/// takes (`layout`: the recurrence's, none where the model has none): its
/// own rows and what each lane may carry unfolded into it
/// ([`engine::fire::UNFOLDED_ROWS`]). The fire's, outside the pool.
#[must_use]
pub fn rs_scratch(layout: Option<&crate::rs::Layout>, tokens: u32, lanes: u32) -> u64 {
    layout.map_or(0, |layout| {
        let carried = lanes.saturating_mul(engine::fire::UNFOLDED_ROWS);
        crate::rs::Seat::scratch_bytes(layout, tokens.saturating_add(carried), lanes)
    })
}

/// What a fire takes of a device with `left` bytes past the weights: its
/// span's `act` chunks and its replay scratch (`rs` bytes, outside the
/// pool); or [`Probe::Pool`] where the pool laid in what the scratch
/// leaves ([`pool`]) does not hold one working set beside the span.
#[must_use]
pub fn takes(
    left: u64,
    kv_budget: Option<u64>,
    chunk: u64,
    act: u32,
    rs: u64,
    least_chunks: u32,
    binding: u64,
) -> Probe {
    match pool(
        left.saturating_sub(rs),
        kv_budget,
        chunk,
        act,
        least_chunks,
        binding,
    ) {
        Ok(_) => Probe::Takes(u64::from(act).saturating_mul(chunk).saturating_add(rs)),
        Err(_) => Probe::Pool,
    }
}

/// The largest rung a device that takes `workgroups` on each axis of one
/// dispatch can fire: a power of two of tokens, a row a workgroup on the
/// narrowest axis.
#[must_use]
pub fn dispatched_rung(workgroups: [u32; 3]) -> u32 {
    let rows = workgroups.into_iter().min().unwrap_or(1).max(1);
    1 << rows.ilog2()
}

/// What held a fitted figure where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The load states the figure: the fit leaves it as it is.
    Stated,
    /// The model declares it, and it is taken whole.
    Model,
    /// The most rows the device dispatches on an axis.
    DispatchAxis,
    /// The fire's share of what the device has left past the weights: the
    /// next one takes more.
    Share,
    /// The pool the load lays: it does not hold the next one's span beside
    /// one working set.
    Pool,
    /// The plan: the next one is not planned (the compiler does not carve
    /// it, or the store does not lay its rows).
    Plan,
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Stop::Stated => "stated",
            Stop::Model => "model",
            Stop::DispatchAxis => "dispatch axis",
            Stop::Share => "share",
            Stop::Pool => "pool sizing",
            Stop::Plan => "plan",
        })
    }
}

/// What a fire of a candidate rung takes of the device, or why the load
/// does not lay it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    /// Its span's bytes.
    Takes(u64),
    /// [`Stop::Pool`].
    Pool,
    /// [`Stop::Plan`].
    Plan,
}

/// The lowest rung the halving from `dispatched` reaches: the smallest
/// whose fire carries whole the widest run its own ladders admit (`run`,
/// in token rows at a rung: one for a model of text alone, else the widest
/// image, clip or audio item, whose rows no cut divides across fires).
/// `dispatched` itself where no rung under it does.
#[must_use]
pub fn floor(dispatched: u32, run: impl Fn(u32) -> u32) -> u32 {
    let mut tokens = dispatched.max(1);
    while tokens > 1 && run(tokens / 2) <= tokens / 2 {
        tokens /= 2;
    }
    tokens
}

/// The rung a load fires and what held it there: `stated` where the load
/// states one, else the largest at or under `dispatched`, halving and
/// never under `floor` ([`floor`]), whose fire (`probe`) takes no more than
/// `share` and is laid. The floor is not probed: a device that does not
/// hold it is refused by the load, by name.
pub fn rung(
    stated: u32,
    dispatched: u32,
    floor: u32,
    share: u64,
    probe: impl Fn(u32) -> Probe,
) -> (u32, Stop) {
    if stated != 0 {
        return (stated, Stop::Stated);
    }
    // Each refusal halves, so the last one is of the rung next above the
    // one taken.
    let refused = std::cell::Cell::new(Stop::DispatchAxis);
    let floor = floor.clamp(1, dispatched.max(1));
    let tokens =
        engine::fit::tokens_within(dispatched, floor, share, |tokens| match probe(tokens) {
            Probe::Takes(bytes) => {
                if bytes > share {
                    refused.set(Stop::Share);
                }
                bytes
            }
            Probe::Pool => {
                refused.set(Stop::Pool);
                u64::MAX
            }
            Probe::Plan => {
                refused.set(Stop::Plan);
                u64::MAX
            }
        });
    (tokens, refused.get())
}

/// The context a load serves and what held it there: `asked` as it is
/// where the load states it (`declared` false) or its state is budgeted
/// (`fitted` false); else the longest at or under `asked`, in pages of
/// `page` tokens and never under one, at which one sequence's working set
/// (`probe`, `None` where it is not planned) fits `share`.
pub fn context(
    asked: u32,
    declared: bool,
    fitted: bool,
    page: u32,
    share: u64,
    probe: impl Fn(u32) -> Option<u64>,
) -> (u32, Stop) {
    if !declared {
        return (asked, Stop::Stated);
    }
    if !fitted {
        return (asked, Stop::Model);
    }
    let page = u64::from(page.max(1));
    let tokens = |pages: u64| u32::try_from(pages.saturating_mul(page)).unwrap_or(u32::MAX);
    let whole = u64::from(asked).div_ceil(page);
    let pages = engine::fit::pages_within(whole, share, |pages| {
        probe(tokens(pages)).unwrap_or(u64::MAX)
    });
    if pages >= whole {
        return (asked, Stop::Model);
    }
    let stop = match probe(tokens(pages + 1)) {
        Some(_) => Stop::Share,
        None => Stop::Plan,
    };
    // Never under one page: a device with no room for that is refused by
    // the load, by what one working set takes.
    (tokens(pages.max(1)), stop)
}

/// The load's one line on what it fitted.
fn say(
    device: &str,
    left: u64,
    weights: Option<u64>,
    fitted: &LoadBudgets,
    asked: (u32, u32),
    floor: Option<u32>,
    stops: (Stop, Stop),
) {
    tracing::info!(
        device,
        left,
        weights,
        max_tokens = fitted.max_tokens,
        max_lanes = fitted.max_lanes,
        max_context = fitted.max_context,
        asked_tokens = asked.0,
        asked_context = asked.1,
        floor_tokens = floor,
        tokens_stop = %stops.0,
        context_stop = %stops.1,
        "the fire and the context are fitted to the device"
    );
}

/// `budgets` with what the load left for its device to say: the fire's
/// tokens where it states none (and its lanes, no more than them), and the
/// model's declared context shortened to what the device holds. A load
/// that states both is returned as it is.
pub fn budgets<B: Serve>(
    trace: &Trace,
    budgets: LoadBudgets,
    residency: &Residency,
    frames_in_flight: u8,
    boot: &B::DeviceBoot,
    device: &B::Context,
    tuning: &B::Tuning,
) -> Result<LoadBudgets> {
    let fits_tokens = budgets.max_tokens == 0;
    let fits_context = budgets.context_declared && residency.device_kv_budget.is_none();
    if !fits_tokens && !fits_context {
        let (_, tokens_stop) = rung(budgets.max_tokens, 0, 1, 0, |_| Probe::Plan);
        let (_, context_stop) = context(
            budgets.max_context,
            budgets.context_declared,
            false,
            budgets.page_size,
            0,
            |_| None,
        );
        say(
            device.name(),
            device.left(),
            None,
            &budgets,
            (budgets.max_tokens, budgets.max_context),
            None,
            (tokens_stop, context_stop),
        );
        return Ok(budgets);
    }
    let chip = tuning.profile();
    let fused = poem_compiler::fuse::fuse(
        trace.clone(),
        poem_compiler::fuse::flags::on(chip, B::FUSIONS),
    );
    let profile = DeviceProfile::of(chip, device.cores(), &[], &[]);
    let weights: u64 = crate::weights::plane_bytes(&fused)?.iter().sum();
    let after_weights = device.left().saturating_sub(weights);
    let share = tuning.working_share(after_weights);
    let facts = kv::probe(&fused)?;
    let rs_layout = crate::rs::Layout::read(&fused)?;
    let runahead = engine::runahead::Runahead::of(frames_in_flight);

    // What the load plans at `at`, as `Shell::load` does: the compiler's
    // budgets, how the rows lie in the pool's chunks, and one working set.
    let state = |at: &LoadBudgets| -> Result<(Budgets, Rows, Paging, u64)> {
        let base = Budgets::of(crate::api::bake_budgets(at, trace));
        let base = match crate::api::patch_ladder(trace, at) {
            None => base,
            Some(ladder) => base.with_patches(ladder),
        };
        let base = match poem_compiler::VoxelLadder::of(
            trace,
            at.max_voxels,
            at.max_clips,
            at.max_lanes,
        ) {
            None => base,
            Some(ladder) => base.with_voxels(ladder),
        };
        let mut compiler = poem_compiler::fit_voxels(&fused, base, &profile, share);
        let mut slots =
            poem_compiler::state_slots(at.slots, trace, crate::api::DEFAULT_STATE_SLOTS);
        if slots == 0 {
            slots = poem_compiler::fit_frame_slots(
                &fused,
                compiler.voxels.as_mut(),
                share,
                engine::fit::least_state_slots(1),
                runahead.frames_in_flight,
            );
        }
        let recipe = crate::chunks::recipe(
            &fused,
            &compiler,
            at.page_size,
            at.max_context,
            device.max_buffer(),
            B::chunk_bytes(boot),
        )?;
        let rows = Rows::plan(&fused, &recipe, &facts, rs_layout.as_ref())?;
        let page_size = if rows.tokens > 0 {
            rows.tokens
        } else {
            at.page_size
        };
        let asked = Paging::of(page_size, at.max_context, slots, 1)?;
        let least = u64::from(least_chunks(&rows, asked)).saturating_mul(rows.chunk_bytes);
        Ok((compiler, rows, asked, least))
    };
    // What a fire of `at`'s tokens takes of the device: its span, in the
    // whole chunks it takes of the pool's top, and a recurrent model's
    // replay scratch; or why the load does not lay it: the compiler does
    // not carve it, or the load's pool does not hold one working set
    // beside it.
    let fire = |at: &LoadBudgets| -> Probe {
        let planned = || -> Result<(u64, u32, u32)> {
            let (compiler, rows, asked, _) = state(at)?;
            let compiled = compile_axes(&fused, &compiler, &profile)?;
            let least = Paging {
                pages: u64::from(asked.pages_per_slot),
                ..asked
            };
            let span = Span::<B>::plan(
                &fused,
                &compiled,
                &compiler,
                &[at.max_tokens],
                least,
                tuning,
            )?;
            let chunk = rows.chunk_bytes;
            let act = u32::try_from(span.bytes().div_ceil(chunk.max(1))).unwrap_or(u32::MAX);
            Ok((chunk, act, least_chunks(&rows, asked)))
        };
        let Ok((chunk, act, least)) = planned() else {
            return Probe::Plan;
        };
        takes(
            after_weights,
            residency.device_kv_budget,
            chunk,
            act,
            rs_scratch(rs_layout.as_ref(), at.max_tokens, at.max_lanes),
            least,
            device.max_buffer(),
        )
    };

    let asked_lanes = budgets.max_lanes;
    let asked_tokens = if fits_tokens {
        dispatched_rung(device.max_workgroups())
    } else {
        budgets.max_tokens
    };
    let at = |tokens: u32, context: u32| LoadBudgets {
        max_tokens: tokens,
        max_lanes: asked_lanes.min(tokens),
        max_context: context,
        ..budgets.clone()
    };

    let (context, context_stop) = context(
        budgets.max_context,
        budgets.context_declared,
        fits_context,
        budgets.page_size,
        share,
        |context| {
            state(&at(asked_tokens, context))
                .ok()
                .map(|(.., least)| least)
        },
    );
    if context < budgets.max_context {
        tracing::warn!(
            declared = budgets.max_context,
            serving = context,
            "the model declares a context one sequence of which takes more than a fire's \
             share of what the device has left past the weights; serving a shorter one. \
             State `[engine] max_context` to choose."
        );
    }

    // The widest run a fire of `tokens` carries whole, in token rows: one
    // where the model reads text alone, else the widest image the patch
    // ladder admits there, a token row a patch at most (how many patches
    // fold into a token is the package's to state, and the load does not
    // read it). A clip's rows are its request's: the voxel ladder is
    // fitted to memory and states no widest one.
    let run = |tokens: u32| {
        crate::api::patch_ladder(trace, &at(tokens, context)).map_or(1, |ladder| ladder.max_patches)
    };
    let carried = floor(asked_tokens, run);
    // A fire has its share of what the weights leave, as on the cuda
    // shell; a device that does not hold the floor's is refused by the
    // load, by name.
    let (tokens, tokens_stop) = rung(budgets.max_tokens, asked_tokens, carried, share, |tokens| {
        fire(&at(tokens, context))
    });
    let fitted = at(tokens, context);
    say(
        device.name(),
        device.left(),
        Some(weights),
        &fitted,
        (asked_tokens, budgets.max_context),
        fits_tokens.then_some(carried),
        (tokens_stop, context_stop),
    );
    Ok(fitted)
}

#[cfg(test)]
mod tests {
    use super::{
        Pool, Probe, Stop, context, dispatched_rung, floor, pool, rs_scratch, rung, takes,
    };

    const MIB: u64 = 1 << 20;
    /// A share no fixture's fire reaches.
    const ROOMY: u64 = 1 << 40;

    /// A rung names what held it: the load's statement, the device's
    /// dispatch axis where the asked rung is taken whole, and where it is
    /// halved, why the rung next above was not taken.
    #[test]
    fn a_rung_names_what_stopped_it() {
        let takes = |tokens: u32| Probe::Takes(u64::from(tokens) * 1024);
        assert_eq!(rung(10_240, 32_768, 1, 0, takes), (10_240, Stop::Stated));
        assert_eq!(
            rung(0, 32_768, 1, ROOMY, takes),
            (32_768, Stop::DispatchAxis)
        );
        assert_eq!(rung(0, 32_768, 1, 5000 * 1024, takes), (4096, Stop::Share));
        let pooled = |tokens: u32| {
            if tokens > 8192 {
                Probe::Pool
            } else {
                takes(tokens)
            }
        };
        assert_eq!(rung(0, 32_768, 1, ROOMY, pooled), (8192, Stop::Pool));
        let planned = |tokens: u32| {
            if tokens > 2048 {
                Probe::Plan
            } else {
                takes(tokens)
            }
        };
        assert_eq!(rung(0, 32_768, 1, ROOMY, planned), (2048, Stop::Plan));
        // The share stops a rung the pool would have laid, and the last
        // refusal is the one named.
        let both = |tokens: u32| {
            if tokens > 16_384 {
                Probe::Pool
            } else {
                takes(tokens)
            }
        };
        assert_eq!(rung(0, 32_768, 1, 5000 * 1024, both), (4096, Stop::Share));
        // A device that holds no rung is left at one token for the load to
        // refuse.
        assert_eq!(rung(0, 32_768, 1, 0, takes), (1, Stop::Share));
    }

    /// A context names what held it: the load's statement, the model where
    /// its declared context is taken whole (fitted or budgeted), and where
    /// it is shortened, why the next page was not taken.
    #[test]
    fn a_context_names_what_stopped_it() {
        let takes = |context: u32| Some(u64::from(context) * 1000);
        assert_eq!(
            context(4096, false, true, 16, 0, takes),
            (4096, Stop::Stated)
        );
        assert_eq!(
            context(40_960, true, false, 16, 0, takes),
            (40_960, Stop::Model)
        );
        assert_eq!(
            context(40_960, true, true, 16, ROOMY, takes),
            (40_960, Stop::Model)
        );
        assert_eq!(
            context(40_960, true, true, 16, 8200 * 1000, takes),
            (8192, Stop::Share)
        );
        let planned = |context: u32| if context > 4096 { None } else { takes(context) };
        assert_eq!(
            context(40_960, true, true, 16, ROOMY, planned),
            (4096, Stop::Plan)
        );
        assert_eq!(context(40_960, true, true, 16, 0, takes), (16, Stop::Share));
    }

    /// A hybrid whose replay scratch is ten times its span, on a device
    /// too small for the asked rung: the rung is halved to the widest whose
    /// span and scratch fit the share, and the pool the load lays beside
    /// them holds its working set, so the load is not refused.
    #[test]
    fn a_replay_scratch_ten_times_the_span_halves_the_rung_and_the_load_stands() {
        const CHUNK: u64 = 2 * MIB;
        const BINDING: u64 = 4096 * MIB;
        // A token: 64 KiB of span, ten times that of replay scratch; one
        // working set of 64 chunks; 8 GiB past the weights, half the
        // fire's.
        let left = 8192 * MIB;
        let share = left / 2;
        let act =
            |tokens: u32| u32::try_from((u64::from(tokens) * 65_536).div_ceil(CHUNK)).unwrap();
        let rs = |tokens: u32| u64::from(tokens) * 655_360;
        let fire = |tokens: u32| takes(left, None, CHUNK, act(tokens), rs(tokens), 64, BINDING);

        // The asked rung's scratch alone is past the device: no pool.
        assert!(rs(32_768) > left);
        assert_eq!(fire(32_768), Probe::Pool);
        // 8192 tokens are laid and take more than the share (5632 MiB);
        // 4096 take 2816 MiB of it.
        assert_eq!(fire(8192), Probe::Takes(5632 * MIB));
        assert_eq!(rung(0, 32_768, 1, share, fire), (4096, Stop::Share));
        // The load at the fitted rung: the pool in what the scratch leaves
        // holds the span and far more than one working set.
        let laid = pool(left - rs(4096), None, CHUNK, act(4096), 64, BINDING).unwrap();
        assert_eq!(act(4096), 128);
        assert_eq!(laid.total, 2816);
        assert!(laid.total - act(4096) >= 64);
        // The span alone would have fitted at the asked rung: the scratch
        // is what the halving saw.
        let spans = |tokens: u32| takes(left, None, CHUNK, act(tokens), 0, 64, BINDING);
        assert_eq!(
            rung(0, 32_768, 1, share, spans),
            (32_768, Stop::DispatchAxis)
        );
    }

    /// A fire within its share that leaves the pool no working set is
    /// halved further, never refused; a device that seats the working set
    /// beside no fire is left at one token for the load to refuse by name.
    #[test]
    fn a_fire_that_leaves_no_state_is_halved_to_one_that_does() {
        const CHUNK: u64 = 2 * MIB;
        const BINDING: u64 = 4096 * MIB;
        let left = 1024 * MIB;
        // A working set of 340 chunks (680 MiB); a token's span 16 KiB, its
        // scratch 160 KiB.
        let act =
            |tokens: u32| u32::try_from((u64::from(tokens) * 16_384).div_ceil(CHUNK)).unwrap();
        let rs = |tokens: u32| u64::from(tokens) * 163_840;
        let fire = |tokens: u32| takes(left, None, CHUNK, act(tokens), rs(tokens), 340, BINDING);
        // 2048 tokens: 352 MiB, within a share of 512 MiB, and 704 MiB of
        // pool, 16 chunks of it the span's: 4 short of the working set's 340.
        assert_eq!(fire(2048), Probe::Pool);
        assert_eq!(fire(1024), Probe::Takes(176 * MIB));
        assert_eq!(rung(0, 32_768, 1, left / 2, fire), (1024, Stop::Pool));
        let none = |tokens: u32| {
            takes(
                512 * MIB,
                None,
                CHUNK,
                act(tokens),
                rs(tokens),
                340,
                BINDING,
            )
        };
        assert_eq!(rung(0, 32_768, 1, 256 * MIB, none), (1, Stop::Pool));
        assert!(pool(512 * MIB - rs(1), None, CHUNK, act(1), 340, BINDING).is_err());
    }

    /// A model whose fire carries a run of rows whole is halved no lower
    /// than the smallest rung that holds the run: one token for a model of
    /// text alone, and where the ladder follows the rung, the smallest rung
    /// its own widest item fits. A device that does not seat that rung is
    /// refused by the load, by name.
    #[test]
    fn a_rung_is_halved_no_lower_than_the_run_it_carries_whole() {
        const CHUNK: u64 = 2 * MIB;
        const BINDING: u64 = 4096 * MIB;
        assert_eq!(floor(32_768, |_| 1), 1);
        assert_eq!(floor(32_768, |_| 600), 1024);
        assert_eq!(floor(32_768, |_| 1024), 1024);
        assert_eq!(floor(32_768, |_| 1025), 2048);
        // A ladder that follows the rung down to a least item of 64 rows.
        assert_eq!(floor(32_768, |tokens| tokens.clamp(64, 4096)), 64);
        // A run no rung holds leaves the asked one.
        assert_eq!(floor(32_768, |_| 100_000), 32_768);

        // The device of `a_fire_that_leaves_no_state_is_halved_to_one_that_does`:
        // a model of text alone is halved to 1024 tokens there, and so is one
        // that carries 600 rows whole.
        let left = 1024 * MIB;
        let act =
            |tokens: u32| u32::try_from((u64::from(tokens) * 16_384).div_ceil(CHUNK)).unwrap();
        let rs = |tokens: u32| u64::from(tokens) * 163_840;
        let fire = |tokens: u32| takes(left, None, CHUNK, act(tokens), rs(tokens), 340, BINDING);
        let share = left / 2;
        let carried = floor(32_768, |_| 600);
        assert_eq!(rung(0, 32_768, carried, share, fire), (1024, Stop::Pool));
        // One that carries 1500 rows whole is halved no lower than 2048,
        // which this device does not seat: the load refuses it, by name.
        let carried = floor(32_768, |_| 1500);
        assert_eq!(carried, 2048);
        assert_eq!(rung(0, 32_768, carried, share, fire), (2048, Stop::Pool));
        let refused = pool(left - rs(2048), None, CHUNK, act(2048), 340, BINDING)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("state pool"), "{refused}");
    }

    /// A recurrent model's scratch is priced at the rung's own rows and
    /// what every lane may carry unfolded; a model with no recurrence has
    /// none.
    #[test]
    fn a_replay_scratch_is_priced_at_the_rungs_rows_and_what_its_lanes_carry() {
        use crate::rs::{Layout, Region, Seat};
        let layout = Layout {
            regions: vec![Region {
                width: 1024,
                dtype: poem_ir::Dtype::F32,
                row_bytes: 4096,
            }],
            ..Layout::default()
        };
        let carried = 64 * engine::fire::UNFOLDED_ROWS;
        assert_eq!(
            rs_scratch(Some(&layout), 8192, 64),
            Seat::scratch_bytes(&layout, 8192 + carried, 64)
        );
        assert!(rs_scratch(Some(&layout), 8192, 64) >= u64::from(8192 + carried) * 4096);
        assert_eq!(rs_scratch(None, 8192, 64), 0);
    }

    #[test]
    fn a_stop_is_named_in_the_loads_words() {
        let names = [
            (Stop::Stated, "stated"),
            (Stop::Model, "model"),
            (Stop::DispatchAxis, "dispatch axis"),
            (Stop::Share, "share"),
            (Stop::Pool, "pool sizing"),
            (Stop::Plan, "plan"),
        ];
        for (stop, name) in names {
            assert_eq!(stop.to_string(), name);
        }
    }

    /// A span the segments cannot keep whole beside the state the device
    /// has room for is refused by name (the fit halves the rung past it);
    /// half of it is laid, in four segments within one binding each.
    #[test]
    fn a_pool_is_laid_where_its_segments_keep_the_span_whole() {
        let (left, binding) = (15_360 * MIB, 4096 * MIB);
        let refused = pool(left, None, MIB, 4096, 8, binding).expect_err("three segments");
        assert!(
            refused
                .to_string()
                .contains("binds four segments of 4096 MiB"),
            "{refused}"
        );
        let laid = pool(left, None, MIB, 2048, 8, binding).expect("four segments");
        assert_eq!(
            laid,
            Pool {
                room: left,
                total: 8188 + 2048,
                segment_chunks: 2559,
                words: 8188,
            }
        );
    }

    /// One working set stands beside the span or the pool is refused; a
    /// stated state budget sizes the pool beside the span.
    #[test]
    fn a_pool_seats_one_working_set_beside_the_span() {
        let binding = 4096 * MIB;
        assert!(pool(10 * MIB, None, MIB, 8, 3, binding).is_err());
        let laid = pool(64 * MIB, None, MIB, 8, 3, binding).expect("room for both");
        assert_eq!((laid.total, laid.room), (64, 64 * MIB));
        let stated = pool(64 * MIB, Some(16 * MIB), MIB, 8, 3, binding).expect("stated");
        assert_eq!((stated.total, stated.room), (24, 24 * MIB));
    }

    #[test]
    fn the_asked_rung_is_the_largest_power_of_two_an_axis_dispatches() {
        // Vulkan's floor and what a desktop driver reports; WebGPU's default.
        assert_eq!(dispatched_rung([65_535, 65_535, 65_535]), 32_768);
        assert_eq!(dispatched_rung([2_147_483_647, 65_535, 65_535]), 32_768);
        assert_eq!(dispatched_rung([65_536, 65_536, 65_536]), 65_536);
        assert_eq!(dispatched_rung([256, 256, 64]), 64);
        assert_eq!(dispatched_rung([0, 0, 0]), 1);
    }
}
