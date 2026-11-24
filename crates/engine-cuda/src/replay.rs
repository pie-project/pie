//! Bounded replay (DeepSeek-V4.1's Causal Encoder-Decoder): a long prompt's
//! leading rows run the encoder alone and the decoder replays only its last
//! window. Such a step fires twice: the leading rows as `encode` lanes
//! (unannounced; the replay floor op stamps where decoder windows start), then
//! the step with each prompt cut to its last window.

use engine::fire::{ListKind, Readout, RsReset, RsVerb, Step};
use poem_ir::{Attn, Builtin, Fact, Fused, Operation};

use crate::serve::Shell;
use crate::store::layout::LIST;

/// Where a load replays: the word bit of the encoder-only rows, and the
/// window the decoder replays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Replay {
    pub bit: u64,
    pub window: u32,
}

impl Replay {
    /// The load's replay, where its profile turns it on
    /// (`kernel.sdpa.compressed_replay`) and its trace reads the floor.
    #[must_use]
    pub fn of(shell: &Shell) -> Option<Replay> {
        let replay = Replay::in_trace(shell.trace())?;
        shell.tuning().compressed_replay().then_some(replay)
    }

    /// The replay `trace` states: its `encode` fact and its floored
    /// window's width.
    #[must_use]
    pub fn in_trace(trace: &poem_ir::Trace) -> Option<Replay> {
        let bit = trace
            .facts
            .table
            .iter()
            .position(|fact| *fact == Fact::Builtin(Builtin::Encode))?;
        let floored = |attn: &Attn| match attn {
            Attn::Prefill {
                window,
                mask: None,
                selection: None,
                bias: None,
                lse: Some(_),
                ..
            } => *window,
            _ => None,
        };
        let window = trace.nodes.iter().find_map(|node| match &node.op {
            Operation::Attn(attn) => floored(attn),
            Operation::Fused(Fused::PoolAttention { window, .. }) => floored(window),
            _ => None,
        })?;
        Some(Replay {
            bit: 1u64 << bit,
            window,
        })
    }

    /// `step` as the encoder-only step to fire first and the step to fire
    /// after it, where a lane of it is a prompt longer than the window;
    /// `None` where none is. `plain` says whether a word is a plain lane's, attending one way.
    /// A lane whose list table is a translation (its program places its rows, device
    /// geometry) is never cut.
    /// The lanes of `step` [`Self::split`] cuts, in order: the split's
    /// head lane `i` is lane `cut_lanes[i]` of the step.
    #[must_use]
    pub fn cut_lanes(self, step: &Step, plain: impl Fn(u64) -> bool) -> Vec<u32> {
        (0..step.lanes.len())
            .filter(|&at| self.cuts(step, at, &plain))
            .map(|at| u32::try_from(at).unwrap_or(u32::MAX))
            .collect()
    }

    #[must_use]
    pub fn split(self, step: &Step, plain: impl Fn(u64) -> bool) -> Option<(Step, Step)> {
        let window = self.window as usize;
        let cuts: Vec<Option<usize>> = (0..step.lanes.len())
            .map(|at| {
                self.cuts(step, at, &plain)
                    .then(|| step.lanes[at].tokens.len() - window)
            })
            .collect();
        let count = cuts.iter().flatten().count();
        if count == 0 {
            return None;
        }
        let mut head = Step {
            lanes: Vec::with_capacity(count),
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
            shape: None,
        };
        let mut tail = step.clone();
        for (lane, skipped) in tail.lanes.iter_mut().zip(cuts) {
            let Some(skipped) = skipped else { continue };
            let mut lead = lane.clone();
            lead.word |= self.bit;
            lead.tokens.truncate(skipped);
            lead.readout = Readout::None;
            head.lanes.push(lead);

            lane.tokens.drain(..skipped);
            lane.kv.held += skipped as u32;
            if lane.rs_reset == RsReset::Fresh {
                lane.rs_reset = RsReset::Held;
            }
            if let Readout::Rows(rows) = &mut lane.readout {
                for row in rows {
                    *row -= skipped as u32;
                }
            }
        }
        Some((head, tail))
    }

    /// A plain prompt longer than the window: its program rides the second fire.
    fn cuts(self, step: &Step, at: usize, plain: impl Fn(u64) -> bool) -> bool {
        let lane = &step.lanes[at];
        let window = self.window as usize;
        let rows = lane.tokens.len();
        let at = at as u32;
        rows > window
            && plain(lane.word)
            && lane.positions.is_empty()
            && lane.mask.is_none()
            && lane.adapter.is_none()
            && matches!(lane.rs, RsVerb::Fold)
            && lane.inputs.is_empty()
            && !lane.kv_less
            && lane.list == ListKind::Pages
            && lane
                .tables
                .iter()
                .any(|(shape, refs)| *shape == LIST && !refs.is_empty())
            && lane.group.is_none()
            && match &lane.readout {
                Readout::Rows(read) => read.iter().all(|&row| row as usize >= rows - window),
                Readout::Last | Readout::None => true,
            }
            && !step.media.iter().any(|m| m.lane == at)
            && !step.voxels.iter().any(|v| v.lane == at)
    }
}
