//! Retracting the fires a pipeline wrote past an end.

use super::*;

/// Where one fire began writing one working set.
#[derive(Clone)]
pub(crate) struct SetStart {
    pub(super) model: usize,
    pub(super) engine: crate::engine::EngineId,
    pub(super) kv: crate::store::kv::page_table::WorkingSetId,
    pub(super) cursor: CursorCell,
    pub(super) start: u64,
    /// The set's recurrent state (or windows) the fire's rows land in: its
    /// buffer or its windows' head, taken back with the rows.
    pub(super) rs: Option<crate::store::rs::RsWorkingSetId>,
}

/// What one fire wrote, set by set: `None` where the device placed its rows.
pub(crate) type FireWrites = Option<Vec<SetStart>>;

/// One fire a pipeline keeps for `retract`: what it wrote, and whether its
/// pass stated its rows final (CONTRACT-BC 25), which no retract reaches.
#[derive(Clone)]
pub(crate) struct WrittenFire {
    pub(crate) sets: FireWrites,
    pub(crate) final_rows: bool,
}

pub(super) fn record_writes<C: FireContext>(
    ctx: &mut C,
    pipeline: u32,
    writes: FireWrites,
    final_rows: bool,
) -> Anyhow<()> {
    let pipeline: Resource<Pipeline> = Resource::new_borrow(pipeline);
    let pipeline = ctx.resources().get(&pipeline)?;
    let mut written = pipeline.written.lock().unwrap();
    written.push_back(WrittenFire {
        sets: writes,
        final_rows,
    });
    // A fire further back than the run-ahead window is one the program
    // read past before submitting the newest: never taken back.
    let window = crate::scheduler::run_ahead_window();
    while written.len() > window {
        written.pop_front();
    }
    Ok(())
}

/// The row each working set the last `fires` fires wrote ends at once they
/// are taken back: where the first of them began writing it. `Err` names
/// what cannot be taken back.
pub(super) fn retracted_ends(
    written: &std::collections::VecDeque<WrittenFire>,
    fires: usize,
) -> Result<Vec<SetStart>, String> {
    let mut ends: Vec<SetStart> = Vec::new();
    for (at, fire) in written.iter().skip(written.len() - fires).enumerate() {
        let Some(sets) = &fire.sets else {
            return Err(format!(
                "the {} fire from the end placed its rows where the device says, which the \
                 host cannot take back",
                fires - at
            ));
        };
        for set in sets {
            match ends.iter_mut().find(|end| end.kv == set.kv) {
                Some(end) => end.start = end.start.min(set.start),
                None => ends.push(set.clone()),
            }
        }
    }
    Ok(ends)
}

/// Which of the last `fires` fires (counted from the end, 1 the last) is
/// the latest whose pass stated its rows final, if any.
pub(super) fn final_fire(
    written: &std::collections::VecDeque<WrittenFire>,
    fires: usize,
) -> Option<usize> {
    written
        .iter()
        .rev()
        .take(fires)
        .position(|fire| fire.final_rows)
        .map(|at| at + 1)
}

/// Takes back what the last `fires` fires submitted on the closed pipeline
/// `this` wrote, once every fire on it has landed: each working set they
/// wrote ends where the first of them began writing it. A set whose rows
/// cannot be taken back leaves the pipeline failed by that name, so the set
/// is refused wherever it is used next rather than read past its stop.
pub async fn pipeline_retract<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fires: u32,
) -> Anyhow<Result<(), String>> {
    let fires = fires as usize;
    let (pending, failure, written, ends) = {
        let pipeline = ctx.resources().get(&this)?;
        if !pipeline.scope.is_closed() {
            return Ok(Err(
                "pipeline: retract on an open pipeline: close it first, so no fire follows the \
                 ones taken back"
                    .to_string(),
            ));
        }
        let written = pipeline.written.lock().unwrap();
        if fires > written.len() {
            return Ok(Err(format!(
                "pipeline: retract of {fires} fire(s), past the {} it can take back: the \
                 fires submitted, up to its run-ahead window of {}",
                written.len(),
                crate::scheduler::run_ahead_window()
            )));
        }
        // A fire whose pass stated its rows final is never taken back:
        // refused by name before anything is cut (CONTRACT-BC 25).
        if let Some(at) = final_fire(&written, fires) {
            return Ok(Err(format!(
                "final rows: pipeline: retract of {fires} fire(s) reaches the {at} fire from the \
                 end, whose pass stated its sampled rows final; they are never taken back"
            )));
        }
        (
            pipeline.fires.clone(),
            pipeline.failure.clone(),
            written.len() - fires,
            retracted_ends(&written, fires),
        )
    };
    if fires == 0 {
        return Ok(Ok(()));
    }
    // A fire that may still write is never cut under: every fire on the
    // pipeline lands, and its fold and log settle, first.
    finalize_all(ctx, &pending, false).await?;
    ctx.resources()
        .get(&this)?
        .written
        .lock()
        .unwrap()
        .truncate(written);
    if pipeline_failed(&failure).is_some() {
        return Ok(Ok(()));
    }
    let refused = match ends {
        Err(why) => Some(why),
        Ok(ends) => {
            let mut refused = None;
            for end in &ends {
                let stores = crate::store::registry::get(end.model, end.engine);
                match take_back(&stores, end).await {
                    Ok(()) => retract_cursor(&end.cursor, end.start),
                    Err(why) => {
                        refused = Some(why);
                        break;
                    }
                }
            }
            refused
        }
    };
    if let Some(why) = refused {
        let mut failure = failure.lock().unwrap();
        if failure.is_none() {
            *failure = Some(format!(
                "its rows past where the program stopped could not be taken back: {why}"
            ));
        }
    }
    Ok(Ok(()))
}

/// Returns once kv set `ws` is in no move, waiting on each move's end.
pub(super) async fn kv_out_of_move(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
) {
    loop {
        let ended =
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.moves_ended());
        let notified = ended.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if !crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.set_in_move(ws))
        {
            return;
        }
        notified.await;
    }
}

/// Takes `end`'s set back to where the retracted fires began writing it:
/// its unaccepted run-ahead rows first (refused where the run-ahead bound
/// accepted rows past the cut), then its recurrent state's buffer or its
/// windows' head (refused where a fold reached past the cut), then its KV.
/// Under the lagged fold none of these touches a folded row.
pub(super) async fn take_back(
    stores: &crate::store::registry::Stores,
    end: &SetStart,
) -> Result<(), String> {
    let cut = u32::try_from(end.start).unwrap_or(u32::MAX);
    end.cursor.lock().unwrap().take_back(cut)?;
    if let Some(rs) = end.rs {
        match stores.rs.lock().unwrap().truncate(rs, cut) {
            // A set released already holds nothing to take back.
            Ok(()) | Err(crate::store::rs::RsError::UnknownWorkingSet) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    // A set in a move (a suspend or a sweep pins its pages) is cut once
    // the move ends: cut under it, the pages it freed went back again as
    // the move committed (a kv page released twice).
    loop {
        kv_out_of_move(stores, end.kv).await;
        match crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
            kv.truncate(end.kv, end.start)
        }) {
            Ok(())
            | Err(crate::store::kv::KvStoreError::Table(
                crate::store::kv::page_table::KvTableError::UnknownWorkingSet,
            )) => return Ok(()),
            Err(crate::store::kv::KvStoreError::MoveInFlight) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
}

/// The host's record of a set's rows ends at `end`.
pub(super) fn retract_cursor(cell: &CursorCell, end: u64) {
    let end = u32::try_from(end).unwrap_or(u32::MAX);
    let mut cursor = cell.lock().unwrap();
    cursor.tail = Some(end);
    cursor.written = cursor.written.min(end);
    cursor.spans.retain(|&(from, _, _)| from < end);
}

pub async fn pipeline_drop<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    pipeline_close_inner(ctx, &this).await?;
    ctx.resources().delete(this)?;
    Ok(())
}
