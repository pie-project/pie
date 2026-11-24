use super::*;

pub fn frame_size() -> usize {
    thread_local! {
        static FRAME_SIZE: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    FRAME_SIZE.with(|k| *k.get_or_init(|| crate::model::frame_size().max(1) as usize))
}

pub fn channel_capacity() -> usize {
    (crate::model::channel_capacity() as usize).max(2)
}

pub fn kv_page_size() -> u32 {
    thread_local! {
        static PAGE: std::cell::OnceCell<u32> = const { std::cell::OnceCell::new() };
    }
    PAGE.with(|c| *c.get_or_init(crate::model::kv_page_size))
}

/// Even spans of `n` rows, each at most `max` (an output's `max-rows`).
pub fn readout_spans(n: u32, max: u32) -> Vec<(u32, u32)> {
    even_spans(n, max.max(1))
}

fn even_spans(n: u32, cap: u32) -> Vec<(u32, u32)> {
    if n == 0 {
        return Vec::new();
    }
    let cap = cap.min(n).max(1);
    let k = n.div_ceil(cap).max(1);
    let (q, r) = (n / k, n % k);
    let mut out = Vec::with_capacity(k as usize);
    let mut base = 0u32;
    for i in 0..k {
        let end = base + q + u32::from(i < r);
        out.push((base, end));
        base = end;
    }
    debug_assert_eq!(base, n);
    out
}

pub fn submit_frame(on: &Pipeline, slots: &[Option<&ForwardPass>]) -> Result<(), String> {
    let k = frame_size();
    if slots.len() > k {
        return Err(format!(
            "frame holds {} slot(s); model.frame-size() is {k}",
            slots.len()
        ));
    }
    for pass in slots.iter().flatten() {
        pass.attach_program()?;
    }
    if slots.iter().flatten().next().is_none() {
        return Ok(());
    }
    let mut borrows: Vec<Option<&wit_forward::ForwardPass>> =
        slots.iter().map(|slot| slot.map(|p| &p.wit)).collect();
    borrows.resize(k, None);
    wit_forward::submit(&on.wit, &borrows)
}

/// Fires `pass` up to `budget` times, never more, running ahead of
/// `on_token`, whose `i`-th call reads fire `i`'s output. A break keeps the
/// fires through the one it read and takes back the rows of those
/// submitted past it ([`Pipeline::retract`]) when the pass is
/// [`ForwardPass::retractable`]; a final pass's (the default) stay in its
/// sets. The count kept is returned.
pub async fn run_ahead(
    on: &Pipeline,
    pass: &ForwardPass,
    budget: usize,
    on_token: impl AsyncFnMut() -> Result<std::ops::ControlFlow<()>, String>,
) -> Result<usize, String> {
    if budget == 0 {
        return Ok(0);
    }
    // Every slot of a frame is its own fire: the runtime advances the pass's
    // host shadow and its working sets' cursors (recurrent state included)
    // fire by fire, so a mask or a recurrent state fills the frame like any
    // decode. A pass off its model's default path (a canvas redrawn whole, a
    // drafter block) posts one fire at a time: no model served here runs one
    // under run-ahead, so its frame shape has never been measured.
    let r = if pass.switched() { 1 } else { frame_size() };
    let window_fires = crate::model::run_ahead_window() as usize;
    let window_frames = (window_fires / r.max(1)).max(1);

    let final_rows = pass.is_final();
    run_ahead_driven(
        Driver {
            r,
            window_frames,
            budget,
            submit: |live| {
                let slots: Vec<Option<&ForwardPass>> = vec![Some(pass); live];
                submit_frame(on, &slots)
            },
            close: || on.close(),
            // A final pass's rows past a break stay in its sets.
            retract: |fires| {
                if final_rows {
                    Ok(())
                } else {
                    on.retract(fires)
                }
            },
        },
        on_token,
    )
    .await
}

/// What [`run_ahead`] drives: frames of `r` fires, no more than
/// `window_frames` frames unread, `budget` fires in all.
pub(super) struct Driver<S, C, R> {
    pub(super) r: usize,
    pub(super) window_frames: usize,
    pub(super) budget: usize,
    pub(super) submit: S,
    pub(super) close: C,
    pub(super) retract: R,
}

pub(super) async fn run_ahead_driven<S, C, R>(
    mut d: Driver<S, C, R>,
    mut on_token: impl AsyncFnMut() -> Result<std::ops::ControlFlow<()>, String>,
) -> Result<usize, String>
where
    S: FnMut(usize) -> Result<(), String>,
    C: FnMut(),
    R: FnMut(u32) -> Result<(), String>,
{
    use std::ops::ControlFlow;

    let (r, window_frames, budget) = (d.r, d.window_frames, d.budget);
    let mut submitted = 0usize;
    let mut consumed = 0usize;

    let mut submit_one_frame = |submitted: &mut usize| -> Result<(), String> {
        let live = frame_fires(r, budget, *submitted);
        if live == 0 {
            return Ok(());
        }
        (d.submit)(live)?;
        *submitted += live;
        Ok(())
    };

    for _ in 0..window_frames {
        if submitted >= budget {
            break;
        }
        submit_one_frame(&mut submitted)?;
    }

    let mut ended = false;

    if submitted >= budget && !ended {
        (d.close)();
        ended = true;
    }
    while consumed < submitted {
        if on_token().await? == ControlFlow::Break(()) {
            if !ended {
                (d.close)();
            }
            // The fire whose output broke the loop read a token the program
            // took, so its rows stay; those submitted past it are taken back.
            let kept = consumed + 1;
            if submitted > kept {
                (d.retract)(u32::try_from(submitted - kept).unwrap_or(u32::MAX))?;
            }
            return Ok(kept);
        }
        consumed += 1;
        if submitted < budget && submitted - consumed <= (window_frames - 1) * r {
            submit_one_frame(&mut submitted)?;
        }
        if submitted >= budget && !ended {
            (d.close)();
            ended = true;
        }
    }
    if !ended {
        (d.close)();
    }
    Ok(consumed)
}

/// The fires the next frame of a run of `budget` fires, `submitted` of
/// them submitted, holds: a frame of `r`, never past the budget.
fn frame_fires(r: usize, budget: usize, submitted: usize) -> usize {
    r.min(budget.saturating_sub(submitted))
}

pub struct Pipeline {
    pub(super) wit: wit_pipeline::Pipeline,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            wit: wit_pipeline::Pipeline::new(),
        }
    }

    pub fn close(&self) {
        self.wit.close();
    }

    /// Take back what the last `fires` fires submitted on this closed
    /// pipeline wrote, once every fire on it has landed: the rows a program
    /// ran ahead past its stop.
    pub fn retract(&self, fires: u32) -> Result<(), String> {
        self.wit.retract(fires)
    }

    pub fn park(&self) {
        wit_forward::park(&self.wit);
    }

    /// The host's advice to a program that trades compute for latency:
    /// whether no other pipeline is active beside this one, so its fires
    /// batch alone. It may change from one frame to the next.
    #[must_use]
    pub fn speculation_advised(&self) -> bool {
        self.wit.speculation_advised()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline::new()
    }
}
