//! Bounded replay (DeepSeek-V4.1's Causal Encoder-Decoder): a long prompt's
//! leading rows run the encoder alone and the decoder replays only its last
//! window. Such a step fires twice: the leading rows as `encode` lanes
//! (unannounced; the replay floor op stamps where decoder windows start), then
//! the step with each prompt cut to its last window.

use engine::fire::{Readout, RsReset, RsVerb, Step};
use poem_ir::{Attn, Fact, Fused, Operation, Request};

use crate::serve::Shell;

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
        if !shell.tuning().compressed_replay {
            return None;
        }
        Replay::in_trace(shell.trace())
    }

    /// The replay `trace` states: its `encode` fact and its floored
    /// window's width.
    #[must_use]
    pub fn in_trace(trace: &poem_ir::Trace) -> Option<Replay> {
        let bit = trace
            .facts
            .table
            .iter()
            .position(|fact| *fact == Fact::Flag(Request::ENCODE.to_string()))?;
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
    /// `None` where none is. `wide` says whether a word's class attends both
    /// ways.
    #[must_use]
    pub fn split(self, step: &Step, wide: impl Fn(u64) -> bool) -> Option<(Step, Step)> {
        let window = self.window as usize;
        let cuts: Vec<Option<usize>> = (0..step.lanes.len())
            .map(|at| {
                self.cuts(step, at, &wide)
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
        };
        let mut tail = step.clone();
        for (lane, skipped) in tail.lanes.iter_mut().zip(cuts) {
            let Some(skipped) = skipped else { continue };
            let mut lead = lane.clone();
            lead.word |= self.bit;
            lead.tokens.truncate(skipped);
            lead.readout = Readout::None;
            lead.drafts = false;
            lead.channels.clear();
            head.lanes.push(lead);

            lane.tokens.drain(..skipped);
            lane.kv.held += skipped as u32;
            lane.kv.window_copies.clear();
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

    /// Whether lane `at` of `step` is a plain prompt longer than the window (no mask,
    /// scores, draft, conditioning, media or resolved ports; pages stated; readouts
    /// in its last window). Its program and channels ride the second fire.
    fn cuts(self, step: &Step, at: usize, wide: impl Fn(u64) -> bool) -> bool {
        let lane = &step.lanes[at];
        let window = self.window as usize;
        let rows = lane.tokens.len();
        let at = at as u32;
        rows > window
            && lane.word & self.bit == 0
            && lane.positions.is_empty()
            && lane.mask.is_none()
            && lane.adapter.is_none()
            && !lane.captures_scores
            && !lane.block_draft
            && !wide(lane.word)
            && lane.self_cond.is_none()
            && matches!(lane.rs, RsVerb::Fold)
            && lane.ports.is_empty()
            && lane.attn_classes.is_none()
            && !lane.kv_less
            && !lane.kv.pages.is_empty()
            && lane.kv.translation.is_empty()
            && lane.group.is_none()
            && lane.peer.is_none()
            && match &lane.readout {
                Readout::Rows(read) => read.iter().all(|&row| row as usize >= rows - window),
                Readout::Last | Readout::None => true,
            }
            && !step.media.iter().any(|m| m.lane == at)
            && !step.voxels.iter().any(|v| v.lane == at)
    }
}

#[cfg(test)]
mod tests {
    use super::Replay;
    use engine::fire::{KvDelta, Lane, Readout, RsReset, Step};

    fn prompt(slot: u32, rows: u32, held: u32) -> Lane {
        Lane {
            slot,
            word: 0b1,
            tokens: (0..rows).collect(),
            kv: KvDelta {
                held,
                pages: (0..(held + rows).div_ceil(64)).collect(),
                window_copies: vec![(1, 2)],
                ..KvDelta::default()
            },
            rs_reset: RsReset::Fresh,
            ..Lane::default()
        }
    }

    #[test]
    fn a_long_prompt_fires_its_encoder_rows_first_and_its_last_window_after() {
        let replay = Replay {
            bit: 0b100,
            window: 128,
        };
        let mut read = prompt(3, 300, 0);
        read.readout = Readout::Rows(vec![200, 299]);
        let step = Step {
            lanes: vec![
                Lane::decode(1, 0b1, 7, 40),
                prompt(2, 1000, 64),
                prompt(4, 128, 0),
                read,
            ],
            ..Step::default()
        };
        let (head, tail) = replay
            .split(&step, |_| false)
            .expect("two prompts are longer than the window");
        assert_eq!(head.lanes.len(), 2);
        assert!(head.attachments.is_empty());
        let lead = &head.lanes[0];
        assert_eq!((lead.slot, lead.word), (2, 0b101));
        assert_eq!(lead.tokens, (0..872).collect::<Vec<u32>>());
        assert_eq!(lead.kv.held, 64);
        assert_eq!(lead.readout, Readout::None);
        assert_eq!(lead.kv.window_copies, vec![(1, 2)]);
        assert_eq!(head.lanes[1].tokens.len(), 172);

        assert_eq!(tail.lanes.len(), 4);
        assert_eq!(tail.lanes[0], step.lanes[0], "a decode is left as it is");
        assert_eq!(
            tail.lanes[2], step.lanes[2],
            "a prompt of one window is too"
        );
        let cut = &tail.lanes[1];
        assert_eq!(cut.tokens, (872..1000).collect::<Vec<u32>>());
        assert_eq!(cut.kv.held, 64 + 872);
        assert_eq!(cut.word, 0b1);
        assert!(cut.kv.window_copies.is_empty());
        assert_eq!(
            cut.rs_reset,
            RsReset::Held,
            "its state is the one the head wrote"
        );
        assert_eq!(tail.lanes[3].readout, Readout::Rows(vec![28, 127]));
    }

    #[test]
    fn a_prompt_reading_rows_its_decoder_skips_is_left_whole() {
        let replay = Replay {
            bit: 0b100,
            window: 128,
        };
        let mut read = prompt(3, 300, 0);
        read.readout = Readout::Rows(vec![100, 299]);
        let step = Step {
            lanes: vec![read],
            ..Step::default()
        };
        assert!(replay.split(&step, |_| false).is_none());
    }
}
