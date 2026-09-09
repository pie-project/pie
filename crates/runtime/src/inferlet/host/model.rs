use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::model;
use anyhow::Result;

pub fn lane_stream(stream: models::Stream) -> pie::inferlet::model::LaneStream {
    use pie::inferlet::model::LaneStream;
    match stream {
        models::Stream::Text => LaneStream::Text,
        models::Stream::Image => LaneStream::Image,
        models::Stream::Video => LaneStream::Video,
        models::Stream::Audio => LaneStream::Audio,
        models::Stream::Context => LaneStream::Context,
        models::Stream::Reference => LaneStream::Reference,
    }
}

pub fn catalog_stream(stream: pie::inferlet::model::LaneStream) -> models::Stream {
    use pie::inferlet::model::LaneStream;
    match stream {
        LaneStream::Text => models::Stream::Text,
        LaneStream::Image => models::Stream::Image,
        LaneStream::Video => models::Stream::Video,
        LaneStream::Audio => models::Stream::Audio,
        LaneStream::Context => models::Stream::Context,
        LaneStream::Reference => models::Stream::Reference,
    }
}

fn axis_role(role: models::AxisRole) -> pie::inferlet::model::AxisRole {
    use pie::inferlet::model::AxisRole;
    match role {
        models::AxisRole::Time => AxisRole::Time,
        models::AxisRole::Height => AxisRole::Height,
        models::AxisRole::Width => AxisRole::Width,
        models::AxisRole::Index => AxisRole::Index,
    }
}

fn reading_fact(reading: &models::ReadingFact) -> pie::inferlet::model::ReadingFact {
    use pie::inferlet::model::{PortKind, ReadoutKind};
    pie::inferlet::model::ReadingFact {
        name: reading.name.to_string(),
        index: reading.index,
        has_kv: reading.has_kv,
        takes_tokens: reading.takes_tokens,
        streams: reading.streams.iter().copied().map(lane_stream).collect(),
        ports: reading
            .ports
            .iter()
            .map(|port| pie::inferlet::model::PortFact {
                name: port.name.to_string(),
                kind: match port.kind {
                    models::PortKind::Latents => PortKind::Latents,
                    models::PortKind::LaneVector => PortKind::LaneVector,
                    models::PortKind::Context => PortKind::Context,
                    models::PortKind::AxisPositions => PortKind::AxisPositions,
                    models::PortKind::Voxels => PortKind::Voxels,
                },
                width: port.width,
                dtype: pie::inferlet::types::Dtype::F32,
                streams: port.streams.iter().copied().map(lane_stream).collect(),
                rows: port.rows,
            })
            .collect(),
        positions: reading.positions.as_ref().map(|convention| {
            pie::inferlet::model::PositionConvention {
                axes: convention.axes.iter().copied().map(axis_role).collect(),
                text_axis: convention.text_axis,
                text_origin: convention.text_origin,
                image_follows_text: convention.image_follows_text,
                reference_stride: convention.reference_stride,
            }
        }),
        readout: match reading.readout {
            models::ReadoutKind::Logits => ReadoutKind::Logits,
            models::ReadoutKind::Velocity => ReadoutKind::Velocity,
            models::ReadoutKind::Hidden => ReadoutKind::Hidden,
            models::ReadoutKind::Pixels => ReadoutKind::Pixels,
        },
        readout_width: reading.readout_width,
    }
}

impl pie::inferlet::model::Host for ProcessCtx {
    async fn name(&mut self) -> Result<String> {
        Ok(model::model().name().to_string())
    }

    async fn architecture(&mut self) -> Result<String> {
        Ok(model::model().arch_name().to_string())
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
                bidirectional: caps.draft_bidirectional,
                proposals_from: caps.draft_proposals_from,
            }),
        )
    }

    async fn pass_kind(&mut self) -> Result<pie::inferlet::model::ForwardKind> {
        use pie::inferlet::model::ForwardKind;
        let model = model::model();
        if model.diffusion().is_some() {
            return Ok(ForwardKind::Diffusion);
        }
        let has_rs = model.rs_caps().state_size > 0;
        let has_kv = model.kv_page_size() > 0;
        Ok(match (has_kv, has_rs) {
            (_, false) => ForwardKind::Attention,
            (true, true) => ForwardKind::Hybrid,
            (false, true) => ForwardKind::Recurrent,
        })
    }

    async fn canvas(&mut self) -> Result<Option<pie::inferlet::model::CanvasShape>> {
        Ok(model::model()
            .diffusion()
            .map(|d| pie::inferlet::model::CanvasShape {
                length: d.canvas,
                hidden: d.hidden,
                self_cond_taps: d.self_cond_taps,
            }))
    }

    async fn readings(&mut self) -> Result<Vec<pie::inferlet::model::ReadingFact>> {
        Ok(model::model().readings().iter().map(reading_fact).collect())
    }

    async fn latent(&mut self) -> Result<Option<pie::inferlet::model::LatentSpace>> {
        Ok(model::model().generative().and_then(|g| g.latent).map(|l| {
            pie::inferlet::model::LatentSpace {
                channels: l.channels,
                patch_t: l.patch_t,
                patch_h: l.patch_h,
                patch_w: l.patch_w,
                spatial_compression: l.spatial_compression,
                temporal_compression: l.temporal_compression,
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
                    models::ScheduleKind::Flow => ScheduleKind::Flow,
                    models::ScheduleKind::Epsilon => ScheduleKind::Epsilon,
                    models::ScheduleKind::V => ScheduleKind::V,
                },
                shift: s.shift,
                train_steps: s.train_steps,
                boundary: s.boundary,
                pinned_sigmas: s.pinned_sigmas.clone(),
                stream_shifts: s
                    .stream_shifts
                    .iter()
                    .map(|&(stream, shift)| pie::inferlet::model::StreamShift {
                        lane: lane_stream(stream),
                        shift,
                    })
                    .collect(),
            }))
    }

    async fn max_latent_rows(&mut self) -> Result<u32> {
        Ok(model::model().generative().map_or(0, |g| g.max_rows))
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

    async fn max_embed_length(&mut self) -> Result<u32> {
        Ok(crate::engine::get_spec(0)?.limits.max_forward_tokens as u32)
    }

    async fn prefill_chunk_hint(&mut self) -> Result<u32> {
        let budget = crate::engine::get_spec(0)?.limits.max_forward_tokens;
        let live = crate::inferlet::process::live_count().max(1);
        let page = (model::model().kv_page_size() as usize).max(1);
        let share = (budget / live) / page * page;
        Ok(share.clamp(page.min(budget.max(1)), budget.max(1)) as u32)
    }

    async fn rs_state_size(&mut self) -> Result<u64> {
        Ok(model::model().rs_caps().state_size)
    }

    async fn rs_buffer_page_size(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().buffer_page_size)
    }

    async fn rs_fold_granularity(&mut self) -> Result<u32> {
        Ok(model::model().rs_caps().fold_granularity)
    }

    async fn arena_block_size(&mut self) -> Result<u64> {
        Ok(crate::store::registry::get(0, 0).kv_page_size as u64)
    }
}
