use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::model;
use anyhow::Result;

fn readout(readout: poem_ir::Readout) -> pie::inferlet::model::Readout {
    use pie::inferlet::model::ReadoutKind;
    pie::inferlet::model::Readout {
        kind: match readout.kind {
            poem_ir::ReadoutKind::Logits => ReadoutKind::Logits,
            poem_ir::ReadoutKind::Velocity => ReadoutKind::Velocity,
            poem_ir::ReadoutKind::Hidden => ReadoutKind::Hidden,
            poem_ir::ReadoutKind::Pixels => ReadoutKind::Pixels,
        },
        width: readout.width,
    }
}

/// An i32 input is fed i32, any float one f32.
pub(crate) fn input_decl(input: &poem_ir::InputDecl) -> pie::inferlet::model::InputDecl {
    use pie::inferlet::model::{InputAxis, RowCount};
    pie::inferlet::model::InputDecl {
        name: input.name.clone(),
        axis: match input.axis {
            poem_ir::InputAxis::Lane => InputAxis::Lane,
            poem_ir::InputAxis::Row => InputAxis::Row,
            poem_ir::InputAxis::Voxel => InputAxis::Voxel,
        },
        width: input.width,
        dtype: match input.dtype {
            poem_ir::Dtype::I32 => pie::inferlet::types::Dtype::I32,
            _ => pie::inferlet::types::Dtype::F32,
        },
        rows: input.rows.map(|rows| match rows {
            poem_ir::RowCount::Exact(n) => RowCount::Exact(n),
            poem_ir::RowCount::Multiple(n) => RowCount::Multiple(n),
        }),
    }
}

impl pie::inferlet::model::Host for ProcessCtx {
    async fn name(&mut self) -> Result<String> {
        Ok(model::model().name().to_string())
    }

    async fn architecture(&mut self) -> Result<String> {
        Ok(model::model().arch().to_string())
    }

    async fn default_system_speculation(&mut self) -> Result<bool> {
        Ok(false)
    }

    async fn mtp_depth(&mut self) -> Result<u32> {
        Ok(model::model().eta_caps().mtp_depth)
    }

    async fn draft_block(&mut self) -> Result<Option<pie::inferlet::model::BlockDrafter>> {
        let caps = model::model().eta_caps();
        Ok(
            (caps.draft_block > 0).then_some(pie::inferlet::model::BlockDrafter {
                rows: caps.draft_block,
                mask_token: caps.draft_mask_token,
                proposals_from: caps.draft_proposals_from,
            }),
        )
    }

    async fn switches(&mut self) -> Result<Vec<pie::inferlet::model::SwitchDecl>> {
        Ok(model::model()
            .facts()
            .switches
            .iter()
            .map(|decl| pie::inferlet::model::SwitchDecl {
                name: decl.name.clone(),
                values: decl.values.clone(),
            })
            .collect())
    }

    async fn lane(
        &mut self,
        switches: Vec<(String, String)>,
    ) -> Result<std::result::Result<pie::inferlet::model::LaneDecl, String>> {
        let model = model::model();
        let lane = match model.lane(&switches) {
            Ok(lane) => lane,
            Err(error) => return Ok(Err(error)),
        };
        Ok(Ok(pie::inferlet::model::LaneDecl {
            inputs: lane
                .inputs
                .iter()
                .map(|&id| input_decl(&model.inputs()[id as usize]))
                .collect(),
            state: lane.reads_kv || lane.reads_state,
            embeds: lane.embeds,
            readout: lane.readout.map(readout),
        }))
    }

    async fn latent(&mut self) -> Result<Option<pie::inferlet::model::LatentSpace>> {
        Ok(model::model().generative().and_then(|g| g.latent).map(|l| {
            pie::inferlet::model::LatentSpace {
                channels: l.channels,
                spatial_compression: l.spatial_compression,
                temporal_compression: l.temporal_compression,
                first_frames: l.first_frames,
            }
        }))
    }

    async fn schedule(&mut self) -> Result<Option<pie::inferlet::model::ScheduleFact>> {
        use pie::inferlet::model::ScheduleKind;
        Ok(model::model()
            .generative()
            .and_then(|g| g.schedule.as_ref())
            .map(|s| pie::inferlet::model::ScheduleFact {
                kind: match s.kind {
                    poem::generative::ScheduleKind::Flow => ScheduleKind::Flow,
                    poem::generative::ScheduleKind::Epsilon => ScheduleKind::Epsilon,
                    poem::generative::ScheduleKind::V => ScheduleKind::V,
                },
                shift: s.shift,
                train_steps: s.train_steps,
                boundary: s.boundary,
                pinned_sigmas: s.pinned_sigmas.clone(),
                stream_shifts: s
                    .stream_shifts
                    .iter()
                    .map(|(stream, shift)| pie::inferlet::model::StreamShift {
                        lane: stream.clone(),
                        shift: *shift,
                    })
                    .collect(),
            }))
    }

    async fn max_latent_rows(&mut self) -> Result<u32> {
        let limits = crate::engine::get_spec(0)?.limits;
        Ok(u32::try_from(limits.max_forward_tokens).unwrap_or(u32::MAX))
    }

    async fn output_vocab_size(&mut self) -> Result<u32> {
        Ok(model::model().vocab_size())
    }

    async fn kv_page_size(&mut self) -> Result<u32> {
        Ok(model::model().kv_page_size())
    }

    async fn frame_size(&mut self) -> Result<u32> {
        Ok(crate::scheduler::configured_frame_size() as u32)
    }

    async fn submit_deadline_us(&mut self) -> Result<u64> {
        Ok(crate::scheduler::configured_submit_deadline().as_micros() as u64)
    }

    async fn channel_capacity(&mut self) -> Result<u32> {
        Ok(crate::scheduler::channel_capacity() as u32)
    }

    async fn run_ahead_window(&mut self) -> Result<u32> {
        Ok(crate::scheduler::run_ahead_window() as u32)
    }

    async fn max_readout_rows(&mut self, kind: pie::inferlet::model::ReadoutKind) -> Result<u32> {
        let limits = crate::engine::get_spec(0)?.limits;
        Ok(u32::try_from(readout_rows(limits, kind)).unwrap_or(u32::MAX))
    }

    async fn rs_state_size(&mut self) -> Result<u64> {
        Ok(model::model().rs_caps().state_size)
    }
}

fn readout_rows(
    limits: crate::engine::SchedulerLimits,
    kind: pie::inferlet::model::ReadoutKind,
) -> usize {
    match kind {
        pie::inferlet::model::ReadoutKind::Logits if limits.max_readouts > 0 => {
            limits.max_readouts.min(limits.max_forward_tokens)
        }
        _ => limits.max_forward_tokens,
    }
}

/// The rows a prefill chunk may take: the fire's token budget, or fewer when
/// the engine prefers it (Metal, while the Neural Engine shares a fire only
/// up to its rows).
pub(crate) fn prefill_budget(limits: crate::engine::SchedulerLimits) -> usize {
    match limits.prefill_rows {
        0 => limits.max_forward_tokens,
        rows => rows.min(limits.max_forward_tokens),
    }
}

/// A step spends one row on each other live lane's decode, so the fires a
/// prefill is cut into get the rest of the forward budget.
pub(crate) fn prefill_share(budget: usize, live: usize, page: usize) -> u32 {
    let page = page.max(1);
    let share = budget.saturating_sub(live.saturating_sub(1)) / page * page;
    share.clamp(page.min(budget.max(1)), budget.max(1)) as u32
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_prefill_chunk_stops_at_the_rows_the_engine_prefers() {
        let limits = |max_forward_tokens, prefill_rows| crate::engine::SchedulerLimits {
            max_readouts: 0,
            max_scoring_lanes: 0,
            max_forward_requests: 8,
            max_forward_tokens,
            max_page_refs: 64,
            max_context: 0,
            prefill_rows,
        };
        assert_eq!(super::prefill_budget(limits(10240, 0)), 10240);
        assert_eq!(super::prefill_budget(limits(10240, 2048)), 2048);
        assert_eq!(super::prefill_budget(limits(1024, 2048)), 1024);
    }

    #[test]
    fn a_prefill_chunk_keeps_the_budget_the_decoding_lanes_leave() {
        assert_eq!(super::prefill_share(8192, 256, 16), 7936);
        assert_eq!(super::prefill_share(8192, 1, 16), 8192);
        assert_eq!(super::prefill_share(1024, 4096, 16), 16);
    }
}
