//! Pure mapping from a trace container's descriptor ports to a fire
//! request's forward geometry (token family and KV family).

use grammar::brle::RunMask;
use eta_ir::container::{PortSource, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Port, PortMask};
use eta_ir::types::Dtype;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeEnvelope {
    pub token_count: u32,
    /// Classifier output kept for test observability: `classify_decode_envelope`
    /// derives it, but no production reader consumes it (the wire geometry uses
    /// `token_indptr`).
    #[allow(dead_code)]
    pub lane_count: u32,
    pub token_indptr: Vec<u32>,
    /// Classifier output kept for test observability — see [`Self::lane_count`].
    #[allow(dead_code)]
    pub loop_carried: bool,
    /// `Positions` binds a channel (device-carried) rather than a const —
    /// executing the class demands the positions device port.
    pub device_positions: bool,
    /// `RsFoldLen` binds a channel: the recurrent fold length is the
    /// device's, read off the `rs_fold_len` port at compose — the
    /// speculative round's accepted count (`RsVerb::Window`).
    pub device_fold_len: bool,
}

/// Test-only shape derivation. Production uses the split
/// ([`map_geometry_evaluated_with`]) instead of this all-placeholder
/// template, so the classifier's derived CSRs can be asserted against it.
#[cfg(test)]
impl DecodeEnvelope {
    pub fn template(&self, container: &TraceContainer) -> Result<ReqGeometry, GeometryError> {
        let token_count = self.token_count;
        let qo_indptr = match const_port(container, Port::EmbedIndptr) {
            Some(bytes) => as_u32(Port::EmbedIndptr, bytes)?,
            None => self.token_indptr.clone(),
        };
        let position_ids = match const_port(container, Port::Positions) {
            Some(bytes) => as_u32(Port::Positions, bytes)?,
            None => vec![0; token_count as usize],
        };
        let mut readout_defaulted = false;
        let readout = match const_port(container, Port::Readout) {
            Some(bytes) => as_u32(Port::Readout, bytes)?,
            None => {
                readout_defaulted = true;
                qo_indptr
                    .windows(2)
                    .map(|lane| lane[1].saturating_sub(1))
                    .collect()
            }
        };
        let mut sampling_indices = Vec::with_capacity(readout.len());
        let mut sampling_indptr = Vec::with_capacity(qo_indptr.len());
        sampling_indptr.push(0);
        for lane in qo_indptr.windows(2) {
            for &index in &readout {
                if index >= lane[0] && index < lane[1] {
                    sampling_indices.push(index - lane[0]);
                }
            }
            sampling_indptr.push(sampling_indices.len() as u32);
        }
        if sampling_indices.len() != readout.len() {
            return Err(GeometryError::BadCsr {
                port: Port::Readout,
            });
        }
        Ok(ReqGeometry {
            token_ids: vec![0; token_count as usize],
            position_ids,
            qo_indptr,
            sampling_indptr,
            sampling_indices,
            readout_defaulted,
            ..ReqGeometry::default()
        })
    }
}

/// Pure shape classification of the decode-envelope family. Derivability
/// decides the class; the engine's port mask only decides where it
/// executes — a shape match without device geometry ports falls back to
/// host-evaluated execution rather than erroring.
///
/// Test-only. Production uses the diagnostic form
/// ([`classify_decode_envelope_why`]), which also reports the declining rule.
#[cfg(test)]
pub fn classify_decode_envelope(
    container: &TraceContainer,
) -> Result<Option<DecodeEnvelope>, String> {
    classify_decode_envelope_why(container, &mut String::new())
}

/// `classify_decode_envelope`, plus the rule that declined. `Ok(None)`
/// alone can't distinguish a near-miss from a non-candidate, so every
/// decline below writes `why`, which the production caller logs.
pub fn classify_decode_envelope_why(
    container: &TraceContainer,
    why: &mut String,
) -> Result<Option<DecodeEnvelope>, String> {
    let mut decline = |reason: String| -> Result<Option<DecodeEnvelope>, String> {
        *why = reason;
        Ok(None)
    };
    if !container.externs.is_empty() {
        return decline(format!(
            "the trace has {} extern(s); the class is closed traces only",
            container.externs.len()
        ));
    }
    let channel_for = |port| {
        container
            .ports
            .iter()
            .find_map(|binding| (binding.port == port).then_some(&binding.source))
    };
    let channel_index = |port| match channel_for(port) {
        Some(PortSource::Channel(channel)) => Some(*channel as usize),
        _ => None,
    };
    let Some(token_channel) = channel_index(Port::EmbedTokens) else {
        return decline("EmbedTokens is not bound to a channel".to_string());
    };
    let Some(kv_len_channel) = channel_index(Port::KvLen) else {
        return decline("KvLen is not bound to a channel".to_string());
    };
    let puts_channel = |channel: usize| {
        container.stages.iter().any(|stage| {
            stage
                .ops
                .iter()
                .any(|op| matches!(op, Op::ChanPut { chan, .. } if *chan as usize == channel))
        })
    };
    let loop_carried = puts_channel(token_channel);
    let token = container
        .channels
        .get(token_channel)
        .ok_or_else(|| "decode envelope token channel is out of range".to_string())?;
    let kv_len = container
        .channels
        .get(kv_len_channel)
        .ok_or_else(|| "decode envelope KV-length channel is out of range".to_string())?;
    if (!token.seeded && !loop_carried) || !puts_channel(kv_len_channel) {
        return decline(format!(
            "the token channel must be seeded or loop-carried and the KV-length \
             channel must be put by a stage (token seeded={}, token loop-carried={}, \
             kv-len put={})",
            token.seeded,
            loop_carried,
            puts_channel(kv_len_channel)
        ));
    }
    for port in [
        Port::Positions,
        Port::Pages,
        Port::PageIndptr,
        Port::WSlot,
        Port::WOff,
    ] {
        if channel_for(port).is_none() {
            return decline(format!("{port:?} has no port binding at all"));
        }
    }
    let token_dims = token.shape.dims();
    if token_dims.len() != 1
        || token_dims[0] == 0
        || !matches!(
            token.dtype,
            eta_ir::container::ChanDType::Concrete(Dtype::I32)
                | eta_ir::container::ChanDType::Concrete(Dtype::U32)
        )
    {
        return Err("decode envelope tokens must be a non-empty i32/u32 vector".to_string());
    }
    let token_count = token_dims[0];
    let qo_indptr = match channel_for(Port::EmbedIndptr) {
        None => vec![0, token_count],
        Some(PortSource::Const { dtype, shape, data })
            if *dtype == Dtype::U32 && shape.dims().len() == 1 =>
        {
            if data.len() % 4 != 0 {
                return Err("decode envelope EmbedIndptr has a partial u32".to_string());
            }
            data.chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect()
        }
        Some(PortSource::Channel(channel)) => {
            let declaration = container
                .channels
                .get(*channel as usize)
                .ok_or_else(|| "decode envelope EmbedIndptr channel is out of range".to_string())?;
            let dims = declaration.shape.dims();
            if dims.len() != 1
                || dims[0] < 2
                || dims[0] > token_count + 1
                || !matches!(
                    declaration.dtype,
                    eta_ir::container::ChanDType::Concrete(Dtype::U32)
                )
            {
                return Err(format!(
                    "decode envelope EmbedIndptr channel must be a [lanes + 1] u32 vector with \
                     at most {} lane(s)",
                    token_count
                ));
            }
            if dims[0] == token_count + 1 {
                // One token a lane, whether the split is seeded or a stage
                // puts it: the only CSR of that length.
                (0..=token_count).collect()
            } else if declaration.seeded && !puts_channel(*channel as usize) {
                // A seeded, never-put split: fixed for the pass's life and
                // evaluated off the seed on every fire
                // (`map_geometry_evaluated_with`), so the lanes it cuts are
                // the host's to check there. `[0, tokens]` is the one-lane
                // window a recurrent state can take; a wider seeded split is
                // admitted on the same terms and its CSR is the host's, not
                // this classifier's (`token_indptr` is left empty).
                let lanes = dims[0] - 1;
                if lanes == 1 {
                    vec![0, token_count]
                } else {
                    Vec::new()
                }
            } else {
                return Err(
                    "decode envelope EmbedIndptr put by a stage must declare one token per lane"
                        .to_string(),
                );
            }
        }
        Some(_) => {
            return Err("decode envelope EmbedIndptr must be a u32 vector".to_string());
        }
    };
    // A lane is one token (the decode shape) or a fixed run of them (a
    // speculative WINDOW: the correction and the drafts after it, one lane,
    // `w` rows — the shape a recurrent state can take, since its scan runs a
    // lane's rows in order and a token spread over lanes has no order). The
    // split is a host-known constant either way: what moves between fires is
    // the ids and the geometry, never the row count.
    if !qo_indptr.is_empty()
        && (qo_indptr.len() < 2
            || qo_indptr[0] != 0
            || qo_indptr.last().copied() != Some(token_count)
            || qo_indptr.windows(2).any(|pair| pair[1] <= pair[0]))
    {
        return Err(
            "decode envelope EmbedIndptr must cut the tokens into non-empty lanes, in order"
                .to_string(),
        );
    }
    let lane_count = match channel_for(Port::EmbedIndptr) {
        Some(PortSource::Channel(channel)) => {
            container.channels[*channel as usize].shape.dims()[0] - 1
        }
        _ => (qo_indptr.len() - 1) as u32,
    };
    if kv_len.shape.dims() != [lane_count]
        || !matches!(
            kv_len.dtype,
            eta_ir::container::ChanDType::Concrete(Dtype::U32)
        )
    {
        return Err(format!(
            "decode envelope KV length must be a [{lane_count}] u32 vector"
        ));
    }

    let mut device_positions = false;
    let mut device_fold_len = false;
    for binding in &container.ports {
        match (&binding.port, &binding.source) {
            (Port::EmbedTokens | Port::KvLen, PortSource::Channel(_)) => {}
            // The recurrent fold length on a channel: the device's number
            // (a speculative round's accepted count), one per lane or one
            // for all. A const one is host territory and needs no port.
            (Port::RsFoldLen, PortSource::Channel(channel)) => {
                let declaration = container.channels.get(*channel as usize).ok_or_else(|| {
                    "decode envelope rs_fold_len channel is out of range".to_string()
                })?;
                let dims = declaration.shape.dims();
                if !(dims == [lane_count] || dims == [1])
                    || !matches!(
                        declaration.dtype,
                        eta_ir::container::ChanDType::Concrete(Dtype::U32)
                            | eta_ir::container::ChanDType::Concrete(Dtype::I32)
                    )
                {
                    return Err(format!(
                        "device rs_fold_len must be a [{lane_count}] (or [1]) u32/i32 vector"
                    ));
                }
                device_fold_len = true;
            }
            (Port::RsFoldLen, _) => {}
            (Port::EmbedIndptr, PortSource::Const { dtype, shape, data })
                if *dtype == Dtype::U32
                    && shape.dims() == [lane_count + 1]
                    && data.len() == (lane_count as usize + 1) * 4 => {}
            (Port::EmbedIndptr, PortSource::Channel(channel)) => {
                let declaration = container.channels.get(*channel as usize).ok_or_else(|| {
                    "decode envelope EmbedIndptr channel is out of range".to_string()
                })?;
                if declaration.shape.dims() != [lane_count + 1]
                    || !matches!(
                        declaration.dtype,
                        eta_ir::container::ChanDType::Concrete(Dtype::U32)
                    )
                {
                    return Err("device EmbedIndptr must be a [lanes+1] u32 vector".to_string());
                }
            }
            (Port::Positions, PortSource::Const { dtype, shape, .. })
                if *dtype == Dtype::U32 && shape.dims() == [token_count] => {}
            (Port::Readout, PortSource::Const { dtype, shape, data })
                if *dtype == Dtype::U32
                    && shape.dims().len() == 1
                    && data.len() == shape.dims()[0] as usize * 4
                    && data.chunks_exact(4).all(|bytes| {
                        u32::from_le_bytes(bytes.try_into().unwrap()) < token_count
                    }) => {}
            (Port::Readout, PortSource::Channel(channel)) => {
                let declaration = container
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| "decode envelope Readout channel is out of range".to_string())?;
                if declaration.shape.dims().len() != 1
                    || !matches!(
                        declaration.dtype,
                        eta_ir::container::ChanDType::Concrete(Dtype::U32)
                    )
                {
                    return Err("device Readout must be a u32 vector".to_string());
                }
            }
            (Port::PageIndptr, PortSource::Const { dtype, shape, data })
                if *dtype == Dtype::U32
                    && shape.dims() == [lane_count + 1]
                    && data.len() == (lane_count as usize + 1) * 4 => {}
            (Port::Positions, PortSource::Channel(channel)) => {
                device_positions = true;
                let declaration = container.channels.get(*channel as usize).ok_or_else(|| {
                    "decode envelope position channel is out of range".to_string()
                })?;
                if declaration.shape.dims() != [token_count]
                    || !matches!(
                        declaration.dtype,
                        eta_ir::container::ChanDType::Concrete(Dtype::U32)
                    )
                {
                    return Err(format!(
                        "device-carried positions must be a [{token_count}] u32 vector"
                    ));
                }
            }
            (Port::Pages, PortSource::Channel(channel)) => {
                let declaration = container
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| "decode envelope pages channel is out of range".to_string())?;
                let dims = declaration.shape.dims();
                let valid_shape = match dims {
                    [flat] => *flat > 0,
                    [lanes, stride] => *lanes == lane_count && *stride > 0,
                    _ => false,
                };
                if !valid_shape
                    || !matches!(
                        declaration.dtype,
                        eta_ir::container::ChanDType::Concrete(Dtype::U32)
                    )
                {
                    return Err(
                        "device pages must be a non-empty flat or [lanes,pages] u32 pool"
                            .to_string(),
                    );
                }
            }
            (Port::PageIndptr, PortSource::Channel(channel)) => {
                let declaration = container.channels.get(*channel as usize).ok_or_else(|| {
                    "decode envelope page-indptr channel is out of range".to_string()
                })?;
                if declaration.shape.dims() != [lane_count + 1]
                    || !matches!(
                        declaration.dtype,
                        eta_ir::container::ChanDType::Concrete(Dtype::U32)
                    )
                {
                    return Err("device PageIndptr must be a [lanes+1] u32 vector".to_string());
                }
            }
            (Port::WSlot | Port::WOff, PortSource::Channel(channel)) => {
                let declaration = container
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| "decode envelope write channel is out of range".to_string())?;
                if declaration.shape.dims() != [token_count]
                    || !matches!(
                        declaration.dtype,
                        eta_ir::container::ChanDType::Concrete(Dtype::U32)
                    )
                {
                    return Err("device WSlot/WOff must be a [tokens] u32 vector".to_string());
                }
            }
            (Port::AttnMask, PortSource::Channel(channel)) => {
                // Not this class: a channel-bound dense mask belongs to the
                // pool-owned device-geometry class.
                let declaration = container
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| "decode envelope mask channel is out of range".to_string())?;
                if !matches!(
                    declaration.dtype,
                    eta_ir::container::ChanDType::Concrete(Dtype::Bool)
                ) {
                    return Err("device attention mask must be a bool channel".to_string());
                }
                return decline(
                    "a channel-bound dense AttnMask belongs to the pool-owned \
                     device-geometry class; the envelope compose carries no \
                     per-lane mask state"
                        .to_string(),
                );
            }
            (Port::AttnMask, _) => {
                // A host-known (const) mask is wire territory: the host
                // evaluator synthesizes per-row wire masks for it.
                return Ok(None);
            }
            _ => {
                return Err(format!(
                    "decode envelope cannot resolve {:?} from this source",
                    binding.port
                ));
            }
        }
    }
    Ok(Some(DecodeEnvelope {
        token_count,
        lane_count,
        token_indptr: qo_indptr,
        loop_carried,
        device_positions,
        device_fold_len,
    }))
}

/// The device geometry ports executing `envelope` as the DecodeEnvelope
/// class demands of an engine.
pub fn envelope_required_ports(envelope: &DecodeEnvelope) -> PortMask {
    let mut required = PortMask::of(&[Port::EmbedTokens, Port::KvLen]);
    if envelope.device_positions {
        required = required.with(Port::Positions);
    }
    if envelope.device_fold_len {
        required = required.with(Port::RsFoldLen);
    }
    // No AttnMask entry: the classifier declines a channel-bound mask
    // outright, so no envelope reaching here carries one.
    required
}

#[cfg(test)]
fn const_port(container: &TraceContainer, port: Port) -> Option<&[u8]> {
    container.ports.iter().find_map(|binding| {
        if binding.port != port {
            return None;
        }
        match &binding.source {
            PortSource::Const { data, .. } => Some(data.as_slice()),
            PortSource::Channel(_) => None,
        }
    })
}

/// The forward geometry an ETA pass contributes to a `LaunchPlan`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReqGeometry {
    /// Input token ids (from `embed_tokens`).
    pub token_ids: Vec<u32>,
    /// RoPE positions from the required `positions` channel.
    pub position_ids: Vec<u32>,
    /// Per-lane token CSR from the required `embed_indptr` channel.
    pub qo_indptr: Vec<u32>,
    /// KV page slot ids from the required `pages` channel.
    pub kv_page_indices: Vec<u32>,
    /// Per-lane page CSR from the required `page_indptr` channel.
    pub kv_page_indptr: Vec<u32>,
    /// Each lane's readable KV extent after this fire's append — the
    /// `kv_len` port, verbatim. [`KvDelta::held`](engine::KvDelta) (the
    /// extent before the append) is this minus the lane's rows; kept
    /// undivided so the lowering needs no page size.
    pub kv_len: Vec<u32>,
    /// Read-out positions (from `readout`, else the last token of each lane).
    pub sampling_indices: Vec<u32>,
    /// Per-lane read-out CSR.
    pub sampling_indptr: Vec<u32>,
    /// True when `readout` was absent and the last row of each lane was
    /// synthesized as a default. A fold fire samples nothing, so
    /// `rs_plan_for`'s callers drop these synthesized rows for a folding
    /// fire; an explicit readout is left alone so the engine still refuses
    /// it loudly.
    pub readout_defaulted: bool,
}

/// A geometry-mapping failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// A port bound to a channel had no value at fire time (unfilled cell).
    MissingChannelValue { port: Port, channel: u32 },
    /// A port's byte payload isn't a whole number of `u32`s.
    BadPayload { port: Port, bytes: usize },
    /// A trace-constant CSR does not partition its declared rows.
    BadCsr { port: Port },
    /// No `embed_tokens` port — every pass embeds tokens.
    NoEmbed,
}

impl ReqGeometry {
    /// This geometry as the lanes a fire submits. The three CSR pairs
    /// (`token_ids`/`qo_indptr`, `kv_page_indices`/`kv_page_indptr`,
    /// `sampling_indices`/`sampling_indptr`) are cut back into per-lane
    /// form here, since [`Lane`] is the per-lane representation. A lane
    /// naming no page keeps an empty page list, meaning the shell owns
    /// this slot's page table.
    #[must_use]
    pub fn lanes(&self) -> Vec<::engine::Lane> {
        let cut = |values: &[u32], indptr: &[u32], lane: usize| -> Vec<u32> {
            let (Some(&start), Some(&end)) = (indptr.get(lane), indptr.get(lane + 1)) else {
                return Vec::new();
            };
            values
                .get(start as usize..end as usize)
                .unwrap_or_default()
                .to_vec()
        };
        let count = self.qo_indptr.len().saturating_sub(1);
        (0..count)
            .map(|lane| {
                let tokens = cut(&self.token_ids, &self.qo_indptr, lane);
                let positions = cut(&self.position_ids, &self.qo_indptr, lane);
                let pages = cut(&self.kv_page_indices, &self.kv_page_indptr, lane);
                let rows = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
                // The port's extent is after the append; `held` is before
                // it. Saturating subtraction avoids wrapping on a bad geometry.
                let held = self
                    .kv_len
                    .get(lane)
                    .copied()
                    .unwrap_or(rows)
                    .saturating_sub(rows);
                let readout = cut(&self.sampling_indices, &self.sampling_indptr, lane);
                ::engine::Lane {
                    // Placeholder; stamped by `stamp_lane_slots` before the
                    // fire reaches the scheduler.
                    slot: 0,
                    // Placeholder; stamped by `stamp_lane_words`, after
                    // `FireAttnMask::apply_to` cuts the mask onto the lane.
                    word: 0,
                    tokens,
                    // Empty means the natural run `held .. held + rows`.
                    positions: if positions
                        .iter()
                        .enumerate()
                        .all(|(at, &position)| position == held + at as u32)
                    {
                        Vec::new()
                    } else {
                        positions
                    },
                    kv: ::engine::KvDelta {
                        held,
                        pages,
                        translation: Vec::new(),
                    },
                    mask: None,
                    adapter: None,
                    // The ETA port vocabulary has no draft or capture port yet.
                    drafts: false,
                    captures_scores: false,
                    // Stamped by `stamp_denoise` on a denoise pass's lanes.
                    bidirectional: false,
                    self_cond: None,
                    // This runtime predicts no channel cursor, so `Fold` is
                    // the only recurrent verb served.
                    rs: ::engine::RsVerb::Fold,
                    // Matches the engine's old rule (`kv.held == 0` is a
                    // sequence beginning) until RS gets its own reset class.
                    rs_reset: ::engine::RsReset::Inferred,
                    channels: Vec::new(),
                    readout: match readout.as_slice() {
                        [] => ::engine::Readout::None,
                        [only] if *only + 1 == rows => ::engine::Readout::Last,
                        rows => ::engine::Readout::Rows(rows.to_vec()),
                    },
                }
            })
            .collect()
    }

    /// Write this geometry into a request's lanes, leaving everything else
    /// (the recurrent half, the mask, the tickets) intact.
    pub fn apply_to(&self, req: &mut crate::engine::FireRequest) {
        req.lanes = self.lanes();
    }
}

/// Per-fire lowering of the optional attention-mask descriptor. A
/// channel-backed mask is not intrinsically device-resident: whether it's
/// host-known varies per fire, so host-known masks take the wire BRLE path
/// and only genuinely device-derived values select dense device lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FireAttnMask {
    Omitted,
    Host {
        masks: Vec<::engine::Mask>,
        mask_indptr: Vec<u32>,
    },
    Device,
}

impl FireAttnMask {
    /// Cut this fire's mask onto its lanes. The mask CSR is one row per
    /// query: a lane whose span is one row is a restriction of its whole
    /// readable extent and lowers to `Masking::Extent`; a lane whose span
    /// is several rows is a genuinely two-dimensional mask (rows need not
    /// be nested — e.g. windowed prefill) and lowers to `Masking::Rows`,
    /// the CSR's rows carried through in order.
    pub(crate) fn apply_to(
        self,
        request: &mut crate::engine::FireRequest,
    ) -> Result<(), String> {
        match self {
            FireAttnMask::Omitted => {}
            FireAttnMask::Host { masks, mask_indptr } => {
                // Cut the flat `masks` vector back to one masking per lane
                // (`Lane::mask`) using `mask_indptr`.
                for (lane, request_lane) in request.lanes.iter_mut().enumerate() {
                    let (Some(&start), Some(&end)) =
                        (mask_indptr.get(lane), mask_indptr.get(lane + 1))
                    else {
                        continue;
                    };
                    let Some(rows) = masks.get(start as usize..end as usize) else {
                        return Err(format!(
                            "lane {lane}'s attention mask CSR names rows {start}..{end} \
                             of the {} the fire carries",
                            masks.len()
                        ));
                    };
                    request_lane.mask = match rows {
                        // No row: unmasked, not a synthesized all-keeping one.
                        [] => None,
                        // One row is the lane's whole extent.
                        [only] => Some(::engine::Masking::Extent(only.clone())),
                        // Several, parallel to the lane's query rows.
                        rows => Some(::engine::Masking::Rows(rows.to_vec())),
                    };
                }
                request.has_user_mask = true;
                // A decode-shaped custom mask still needs the mask-aware
                // prefill attention path.
                request.single_token_mode = false;
            }
            FireAttnMask::Device => {
                request.has_user_mask = true;
            }
        }
        Ok(())
    }
}

/// Lower an already-evaluated `AttnMask` port into one BRLE row per query.
pub(crate) fn lower_attn_mask_evaluated(
    container: &TraceContainer,
    qo_indptr: &[u32],
    evaluated: &[(Port, Result<eta_compiler::eval::interp::Value, String>)],
) -> Result<FireAttnMask, String> {
    let Some(binding) = container
        .ports
        .iter()
        .find(|binding| binding.port == Port::AttnMask)
    else {
        return Ok(FireAttnMask::Omitted);
    };
    let value = evaluated
        .iter()
        .find_map(|(port, value)| (*port == Port::AttnMask).then_some(value))
        .ok_or_else(|| "attention-mask port was not evaluated".to_string())?;
    let value = match value {
        Ok(value) => value,
        Err(_) if matches!(binding.source, PortSource::Channel(_)) => {
            return Ok(FireAttnMask::Device);
        }
        Err(error) => {
            return Err(format!(
                "attention-mask constant could not be evaluated: {error}"
            ));
        }
    };
    let eta_compiler::eval::interp::Value::Bool(dense) = value else {
        return Err(format!(
            "attention-mask evaluated as {:?}, expected bool",
            value.dtype()
        ));
    };
    if qo_indptr.len() < 2
        || qo_indptr.first().copied() != Some(0)
        || qo_indptr.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err("attention-mask query CSR is malformed".to_string());
    }
    let query_rows = qo_indptr.last().copied().unwrap_or_default() as usize;
    if query_rows == 0 {
        return Err("attention-mask requires at least one query row".to_string());
    }
    if dense.len() % query_rows != 0 {
        return Err(format!(
            "attention-mask has {} cells for {query_rows} query rows",
            dense.len()
        ));
    }
    let stride = dense.len() / query_rows;
    if stride == 0 {
        return Err("attention-mask key stride is empty".to_string());
    }
    let masks = dense
        .chunks_exact(stride)
        .map(|row| {
            let mask = RunMask::from_slice(row);
            ::engine::Mask::new(mask.buffer, mask.total_size)
        })
        .collect();
    Ok(FireAttnMask::Host {
        masks,
        mask_indptr: qo_indptr.to_vec(),
    })
}

/// Evaluate and lower the mask against this fire's host-shadow value oracle.
pub(crate) fn evaluate_attn_mask(
    bound: &eta_ir::validate::BoundTrace,
    known: &mut dyn FnMut(u32) -> Option<eta_compiler::eval::interp::Value>,
    qo_indptr: &[u32],
) -> Result<FireAttnMask, String> {
    if !bound
        .container
        .ports
        .iter()
        .any(|binding| binding.port == Port::AttnMask)
    {
        return Ok(FireAttnMask::Omitted);
    }
    let evaluated = eta_compiler::eval::pareval::eval_descriptor_ports(bound, known)
        .map_err(|blocker| format!("attention-mask evaluation failed: {blocker}"))?
        .into_iter()
        .map(|(port, value)| (port, value.map_err(|blocker| blocker.to_string())))
        .collect::<Vec<_>>();
    lower_attn_mask_evaluated(&bound.container, qo_indptr, &evaluated)
}

/// Per-channel values at fire time: `values[i]` is channel `i`'s current cell
/// bytes (little-endian, per its dtype), or `None` if unfilled.
pub type ChannelValues<'a> = &'a [Option<Vec<u8>>];

/// Per-port evaluation outcomes, recorded alongside a mapped geometry: for each
/// port that was consulted, the value it evaluated to or the reason it declined.
pub type PortEvaluations = Vec<(Port, Result<eta_compiler::eval::interp::Value, String>)>;

/// An evaluated-geometry failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluatedGeometryError {
    /// A required port's value chain passes through device-only state.
    NotDerivable { port: Port, blocker: String },
    /// A derived value violates the wire contract (a real bug, loud).
    BadValue { port: Port, reason: String },
}

impl std::fmt::Display for EvaluatedGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluatedGeometryError::NotDerivable { port, blocker } => {
                write!(f, "{port:?} is not host-derivable: {blocker}")
            }
            EvaluatedGeometryError::BadValue { port, reason } => {
                write!(f, "{port:?} evaluated to an invalid value: {reason}")
            }
        }
    }
}

/// The declared dims of the channel or const a port binds.
fn port_dims(container: &TraceContainer, port: Port) -> Option<Vec<u32>> {
    let binding = container.ports.iter().find(|b| b.port == port)?;
    match &binding.source {
        PortSource::Const { shape, .. } => Some(shape.dims().to_vec()),
        PortSource::Channel(chan) => Some(
            container
                .channels
                .get(*chan as usize)?
                .shape
                .dims()
                .to_vec(),
        ),
    }
}

/// Map a pass's descriptor ports to forward geometry by evaluating the
/// geometry prologue over host-known channel values — the general form of
/// `map_geometry`. Returns the geometry plus every port's evaluated value.
///
/// A rank-2 `Pages` envelope (`[lanes, P]`) is compacted to the wire CSR by
/// each lane's live page count from `PageIndptr`; rank-1 pages pass through
/// flat.
///
/// `device_resolved` names ports whose blocker is not an error: each is
/// left as a placeholder of the right length instead of refused.
///
/// # Errors
///
/// [`EvaluatedGeometryError`], minus the ports in `device_resolved`.
pub fn map_geometry_evaluated_with(
    bound: &eta_ir::validate::BoundTrace,
    known: &mut dyn FnMut(u32) -> Option<eta_compiler::eval::interp::Value>,
    device_resolved: PortMask,
) -> Result<(ReqGeometry, PortEvaluations), EvaluatedGeometryError> {
    use eta_compiler::eval::interp::Value;

    let container = &bound.container;
    let ports =
        eta_compiler::eval::pareval::eval_descriptor_ports(bound, known).map_err(|blocker| {
            EvaluatedGeometryError::BadValue {
                port: Port::EmbedTokens,
                reason: blocker.to_string(),
            }
        })?;
    let port_value = |port: Port| -> Option<Result<Value, String>> {
        ports.iter().find_map(|(p, slot)| {
            (*p == port).then(|| slot.clone().map_err(|blocker| blocker.to_string()))
        })
    };
    let required_u32 = |port: Port| -> Result<Vec<u32>, EvaluatedGeometryError> {
        match port_value(port) {
            Some(Ok(value)) => Ok(value_as_u32(&value)),
            Some(Err(blocker)) => Err(EvaluatedGeometryError::NotDerivable { port, blocker }),
            None => Err(EvaluatedGeometryError::BadValue {
                port,
                reason: "port is not bound".to_string(),
            }),
        }
    };
    let optional_u32 = |port: Port| -> Result<Option<Vec<u32>>, EvaluatedGeometryError> {
        match port_value(port) {
            Some(Ok(value)) => Ok(Some(value_as_u32(&value))),
            Some(Err(blocker)) => Err(EvaluatedGeometryError::NotDerivable { port, blocker }),
            None => Ok(None),
        }
    };

    // The CSR first: it sizes any placeholder.
    let qo_indptr = required_u32(Port::EmbedIndptr)?;
    let spanned_rows = qo_indptr.last().copied().unwrap_or(0) as usize;
    let token_ids = match required_u32(Port::EmbedTokens) {
        Ok(ids) => ids,
        Err(EvaluatedGeometryError::NotDerivable { port, blocker })
            if device_resolved.contains(Port::EmbedTokens) =>
        {
            let _ = (port, blocker);
            vec![0; spanned_rows]
        }
        Err(error) => return Err(error),
    };
    let mut g = ReqGeometry {
        token_ids,
        qo_indptr,
        ..ReqGeometry::default()
    };

    let kv_len = required_u32(Port::KvLen)?;
    let lanes = g.qo_indptr.len().saturating_sub(1);
    g.position_ids = required_u32(Port::Positions)?;

    // A fire spanning no tokens is a pure replay. The IR has no zero-sized
    // tensor, so its per-token channels carry one unreferenced element,
    // dropped below; elsewhere a mismatch with the CSR stays an error.
    let spanned = g.qo_indptr.last().copied().unwrap_or(0) as usize;
    for (port, tokens) in [
        (Port::EmbedTokens, &mut g.token_ids),
        (Port::Positions, &mut g.position_ids),
    ] {
        if spanned == 0 {
            tokens.clear();
        } else if tokens.len() != spanned {
            return Err(EvaluatedGeometryError::BadValue {
                port,
                reason: format!(
                    "the token CSR spans {spanned} rows but {} were supplied",
                    tokens.len()
                ),
            });
        }
    }

    // Read-out rows distribute over lanes as lane-relative indices. Absent
    // readout samples each lane's last row.
    let readout = match optional_u32(Port::Readout)? {
        Some(readout) => readout,
        None => {
            g.readout_defaulted = true;
            // A lane spanning no rows has no last row to sample.
            g.qo_indptr
                .windows(2)
                .filter(|lane| lane[1] > lane[0])
                .map(|lane| lane[1] - 1)
                .collect()
        }
    };
    let mut sampling_indices = Vec::with_capacity(readout.len());
    let mut sampling_indptr = Vec::with_capacity(g.qo_indptr.len());
    sampling_indptr.push(0);
    for lane in g.qo_indptr.windows(2) {
        for &index in &readout {
            if index >= lane[0] && index < lane[1] {
                sampling_indices.push(index - lane[0]);
            }
        }
        sampling_indptr.push(sampling_indices.len() as u32);
    }
    if sampling_indices.len() != readout.len() {
        return Err(EvaluatedGeometryError::BadValue {
            port: Port::Readout,
            reason: "read-out rows do not partition into the lane CSR".to_string(),
        });
    }
    g.sampling_indices = sampling_indices;
    g.sampling_indptr = sampling_indptr;

    let pages = required_u32(Port::Pages)?;
    let page_indptr = required_u32(Port::PageIndptr)?;
    g.kv_page_indices =
        compact_page_envelope(container, pages, &page_indptr).map_err(|reason| {
            EvaluatedGeometryError::BadValue {
                port: Port::Pages,
                reason,
            }
        })?;
    g.kv_page_indptr = page_indptr;
    if kv_len.len() != lanes {
        return Err(EvaluatedGeometryError::BadValue {
            port: Port::KvLen,
            reason: format!(
                "expected one length for each of {lanes} lanes, got {}",
                kv_len.len()
            ),
        });
    }
    g.kv_len = kv_len.clone();

    let evaluated = ports
        .into_iter()
        .map(|(port, slot)| (port, slot.map_err(|blocker| blocker.to_string())))
        .collect();
    Ok((g, evaluated))
}

/// Reinterpret an evaluated value's lanes as `u32` (i32 tokens bit-cast, the
/// engine's `token_ids` convention; bool as 0/1).
pub(crate) fn value_as_u32(value: &eta_compiler::eval::interp::Value) -> Vec<u32> {
    use eta_compiler::eval::interp::Value;
    match value {
        Value::U32(v) => v.clone(),
        Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
        Value::F32(v) => v.iter().map(|&x| x as u32).collect(),
        Value::Bool(v) => v.iter().map(|&b| b as u32).collect(),
    }
}

/// Compact a `Pages` port value to the wire lane-page CSR: a rank-2
/// `[lanes, P]` envelope (the SDK lowering) keeps each lane's live prefix per
/// `page_indptr`'s counts, mirroring the engine's descriptor resolution;
/// rank-1 pages are already flat and pass through.
pub(crate) fn compact_page_envelope(
    container: &TraceContainer,
    pages: Vec<u32>,
    page_indptr: &[u32],
) -> Result<Vec<u32>, String> {
    let dims = port_dims(container, Port::Pages).unwrap_or_default();
    if dims.len() != 2 {
        let live = page_indptr.last().copied().unwrap_or_default() as usize;
        if live > pages.len() {
            return Err(format!(
                "page CSR claims {live} live pages from a {}-page pool",
                pages.len()
            ));
        }
        return Ok(pages[..live].to_vec());
    }
    let stride = dims[1] as usize;
    let mut compact = Vec::new();
    for (lane, window) in page_indptr.windows(2).enumerate() {
        let count = window[1].saturating_sub(window[0]) as usize;
        if count > stride {
            return Err(format!(
                "lane {lane} claims {count} live pages over a [{},{}] envelope",
                dims[0], dims[1]
            ));
        }
        let row = lane * stride;
        if row + count > pages.len() {
            return Err("page envelope is shorter than its lane CSR".to_string());
        }
        compact.extend_from_slice(&pages[row..row + count]);
    }
    Ok(compact)
}

/// Map a container's ports to the forward geometry, pure. Every
/// descriptor port must resolve to a host-known value here; the
/// device-geometry path does not come through this function — the engine
/// resolves its ports in-graph and the host maps the RESULT through
/// [`map_geometry_evaluated_with`].
pub fn map_geometry(
    container: &TraceContainer,
    values: ChannelValues<'_>,
) -> Result<ReqGeometry, GeometryError> {
    let mut g = ReqGeometry::default();

    // -- token family --
    let tokens = match resolve(container, values, Port::EmbedTokens)? {
        Some(t) => t,
        None => return Err(GeometryError::NoEmbed),
    };
    g.token_ids = as_u32(Port::EmbedTokens, &tokens)?;
    g.qo_indptr = match resolve(container, values, Port::EmbedIndptr)? {
        Some(b) => as_u32(Port::EmbedIndptr, &b)?,
        None => {
            return Err(GeometryError::BadCsr {
                port: Port::EmbedIndptr,
            });
        }
    };
    let lanes = g.qo_indptr.len().saturating_sub(1);

    let kv_len = match resolve(container, values, Port::KvLen)? {
        Some(b) => as_u32(Port::KvLen, &b)?,
        None => return Err(GeometryError::BadCsr { port: Port::KvLen }),
    };
    g.position_ids = match resolve(container, values, Port::Positions)? {
        Some(b) => as_u32(Port::Positions, &b)?,
        None => {
            return Err(GeometryError::BadCsr {
                port: Port::Positions,
            });
        }
    };

    // read-out: explicit positions, else the last token of each lane.
    match resolve(container, values, Port::Readout)? {
        Some(b) => {
            g.sampling_indices = as_u32(Port::Readout, &b)?;
            let n = g.sampling_indices.len() as u32;
            g.sampling_indptr = vec![0, n];
        }
        None => {
            g.sampling_indices = (0..lanes)
                .map(|l| g.qo_indptr[l + 1].saturating_sub(1))
                .collect();
            g.sampling_indptr = (0..=lanes as u32).collect();
        }
    }

    let explicit_pages = resolve(container, values, Port::Pages)?
        .map(|b| as_u32(Port::Pages, &b))
        .transpose()?;
    let explicit_indptr = resolve(container, values, Port::PageIndptr)?
        .map(|b| as_u32(Port::PageIndptr, &b))
        .transpose()?;
    match (explicit_pages, explicit_indptr) {
        (Some(pages), Some(indptr)) => {
            g.kv_page_indices = compact_page_envelope(container, pages, &indptr)
                .map_err(|_| GeometryError::BadCsr { port: Port::Pages })?;
            g.kv_page_indptr = indptr;
        }
        (Some(_), None) => {
            return Err(GeometryError::BadCsr {
                port: Port::PageIndptr,
            });
        }
        (None, _) => return Err(GeometryError::BadCsr { port: Port::Pages }),
    }
    g.kv_len = kv_len.clone();

    Ok(g)
}

/// Resolve a port's value: its const payload, or the current value of the
/// channel it binds. `None` if the container has no such port; a port bound
/// to a channel with no host-known value is an error — the host never
/// guesses a descriptor value.
fn resolve(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    port: Port,
) -> Result<Option<Vec<u8>>, GeometryError> {
    let Some(binding) = container.ports.iter().find(|p| p.port == port) else {
        return Ok(None);
    };
    match &binding.source {
        PortSource::Const { data, .. } => Ok(Some(data.clone())),
        PortSource::Channel(c) => match values.get(*c as usize).and_then(|v| v.clone()) {
            Some(v) => Ok(Some(v)),
            None => Err(GeometryError::MissingChannelValue { port, channel: *c }),
        },
    }
}

/// Reinterpret a little-endian byte payload as `u32`s (4 bytes each). Token ids
/// stored `i32` reinterpret bit-for-bit (the engine's `token_ids` is `u32`).
fn as_u32(port: Port, bytes: &[u8]) -> Result<Vec<u32>, GeometryError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(GeometryError::BadPayload {
            port,
            bytes: bytes.len(),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use eta_ir::op::Op;
    use eta_ir::registry::Stage;
    use eta_ir::types::{Dtype, Shape};

    fn u32_bytes(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
    fn const_port(port: Port, words: &[u32]) -> PortBinding {
        PortBinding {
            port,
            source: PortSource::Const {
                dtype: Dtype::U32,
                shape: Shape::vector(words.len() as u32),
                data: u32_bytes(words),
            },
        }
    }
    fn chan(shape: Shape, dtype: Dtype) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    /// Minimal base fixture; tests add the required explicit geometry channels.
    fn section3_container() -> TraceContainer {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), Dtype::I32), // 0 tok
                chan(Shape::vector(1), Dtype::U32), // 1 len
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                const_port(Port::EmbedIndptr, &[0, 1]),
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(1),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanTake(0)],
            }],
        }
    }

    fn add_explicit_geometry(container: &mut TraceContainer, tokens: u32, lanes: u32) {
        let position = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::vector(tokens), Dtype::U32));
        let pages = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::matrix(lanes, 2), Dtype::U32));
        let page_indptr = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::vector(lanes + 1), Dtype::U32));
        let w_slot = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::vector(tokens), Dtype::U32));
        let w_off = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::vector(tokens), Dtype::U32));
        for (port, channel) in [
            (Port::Positions, position),
            (Port::Pages, pages),
            (Port::PageIndptr, page_indptr),
            (Port::WSlot, w_slot),
            (Port::WOff, w_off),
        ] {
            container.ports.push(PortBinding {
                port,
                source: PortSource::Channel(channel),
            });
        }
    }

    #[test]
    fn section3_single_seq_decode_geometry() {
        let mut c = section3_container();
        add_explicit_geometry(&mut c, 1, 1);
        // tok = [42] (i32), len = [5] (u32); page_size 16.
        let values: Vec<Option<Vec<u8>>> = vec![
            Some(42i32.to_le_bytes().to_vec()),
            Some(5u32.to_le_bytes().to_vec()),
            Some(4u32.to_le_bytes().to_vec()),
            Some([0u32, 0].into_iter().flat_map(u32::to_le_bytes).collect()),
            Some([0u32, 1].into_iter().flat_map(u32::to_le_bytes).collect()),
            Some(0u32.to_le_bytes().to_vec()),
            Some(4u32.to_le_bytes().to_vec()),
        ];
        let g = map_geometry(&c, &values).unwrap();

        assert_eq!(g.token_ids, vec![42]);
        assert_eq!(g.qo_indptr, vec![0, 1], "one lane, one token");
        assert_eq!(
            g.position_ids,
            vec![4],
            "len 5 places the write at position 4"
        );
        assert_eq!(
            g.sampling_indices,
            vec![0],
            "read out the lane's last (only) token"
        );
        assert_eq!(g.sampling_indptr, vec![0, 1]);
        assert_eq!(
            g.kv_len,
            vec![5],
            "the lane's readable extent after the append is 5"
        );
        assert_eq!(g.kv_page_indices, vec![0]);
        assert_eq!(g.kv_page_indptr, vec![0, 1]);
    }

    #[test]
    fn decode_envelope_accepts_shape_equivalent_variants() {
        let mut container = section3_container();
        container.stages[0].ops = vec![
            Op::ChanTake(0),
            Op::ChanPut { chan: 0, value: 0 },
            Op::ChanTake(1),
            Op::ChanPut { chan: 1, value: 1 },
        ];
        add_explicit_geometry(&mut container, 1, 1);
        let envelope = classify_decode_envelope(&container)
            .unwrap()
            .expect("plain loop-carried decode");
        assert_eq!(envelope.token_count, 1);
        assert!(envelope.loop_carried);
        assert!(envelope.device_positions, "channel-fed positions");

        let mut readout = container;
        readout.ports.push(const_port(Port::Readout, &[0]));
        let envelope = classify_decode_envelope(&readout)
            .unwrap()
            .expect("const readout decode");
        assert_eq!(
            envelope.template(&readout).unwrap().sampling_indices,
            vec![0]
        );
    }

    #[test]
    fn decode_envelope_accepts_channel_embed_indptr() {
        let mut container = section3_container();
        container.stages[0].ops = vec![
            Op::ChanTake(0),
            Op::ChanPut { chan: 0, value: 0 },
            Op::ChanTake(1),
            Op::ChanPut { chan: 1, value: 1 },
        ];
        add_explicit_geometry(&mut container, 1, 1);
        let indptr = container.channels.len() as u32;
        container.channels.push(chan(Shape::vector(2), Dtype::U32));
        container
            .ports
            .iter_mut()
            .find(|binding| binding.port == Port::EmbedIndptr)
            .unwrap()
            .source = PortSource::Channel(indptr);

        let envelope = classify_decode_envelope(&container)
            .unwrap()
            .expect("channel indptr decode");
        assert_eq!(envelope.token_indptr, vec![0, 1]);
        assert_eq!(envelope.template(&container).unwrap().qo_indptr, vec![0, 1]);
    }

    #[test]
    fn a_device_carried_bool_mask_is_declined_to_the_pooled_class() {
        let mut container = section3_container();
        container.stages[0].ops = vec![
            Op::ChanTake(0),
            Op::ChanPut { chan: 0, value: 0 },
            Op::ChanTake(1),
            Op::ChanPut { chan: 1, value: 1 },
        ];
        add_explicit_geometry(&mut container, 1, 1);
        let mask = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::matrix(1, 8), Dtype::Bool));
        container.ports.push(PortBinding {
            port: Port::AttnMask,
            source: PortSource::Channel(mask),
        });

        // A masked decode loop is not this class — it is the pool-owned
        // device-geometry class, reached only once the envelope declines.
        let mut why = String::new();
        let classified = classify_decode_envelope_why(&container, &mut why).unwrap();
        assert!(
            classified.is_none(),
            "a channel-bound dense mask must not classify as a decode envelope"
        );
        assert!(
            why.contains("pool-owned device-geometry"),
            "the decline must name where the trace belongs, got {why:?}"
        );

        // A non-bool mask channel is a classification error, not a fallback —
        // checked BEFORE the decline, so a malformed mask is still loud.
        let bad = container.channels.len() as u32 - 1;
        container.channels[bad as usize].dtype = ChanDType::Concrete(Dtype::U32);
        assert!(classify_decode_envelope(&container).is_err());
    }
    #[test]
    fn decode_envelope_accepts_seeded_prefill_tokens() {
        let mut container = section3_container();
        container.channels[0].seeded = true;
        container.stages[0].ops = vec![
            Op::ChanTake(0),
            Op::ChanTake(1),
            Op::ChanPut { chan: 1, value: 1 },
        ];
        add_explicit_geometry(&mut container, 1, 1);

        let envelope = classify_decode_envelope(&container)
            .unwrap()
            .expect("seeded prefill envelope");
        assert!(!envelope.loop_carried);
    }

    #[test]
    fn decode_envelope_derives_multitoken_and_multilane_shapes() {
        let mut multi_token = section3_container();
        multi_token.channels[0].shape = Shape::vector(4);
        multi_token.ports[1] = const_port(Port::EmbedIndptr, &[0, 4]);
        multi_token.stages[0].ops = vec![
            Op::ChanPut { chan: 0, value: 0 },
            Op::ChanPut { chan: 1, value: 1 },
        ];
        add_explicit_geometry(&mut multi_token, 4, 1);
        // One lane of four rows: a speculative window's shape, and the one a
        // recurrent state can take.
        let envelope = classify_decode_envelope(&multi_token).unwrap().unwrap();
        assert_eq!((envelope.token_count, envelope.lane_count), (4, 1));
        assert_eq!(envelope.token_indptr, vec![0, 4]);
        assert_eq!(envelope.template(&multi_token).unwrap().qo_indptr, vec![0, 4]);

        // The same split as a seeded, never-put channel — how a guest states
        // it — classifies the same way; put by a stage it would not.
        let mut seeded_window = section3_container();
        seeded_window.channels[0].shape = Shape::vector(4);
        let split = seeded_window.channels.len() as u32;
        seeded_window.channels.push(chan(Shape::vector(2), Dtype::U32));
        seeded_window.channels[split as usize].seeded = true;
        seeded_window.ports[1] = PortBinding {
            port: Port::EmbedIndptr,
            source: PortSource::Channel(split),
        };
        seeded_window.stages[0].ops = vec![
            Op::ChanPut { chan: 0, value: 0 },
            Op::ChanPut { chan: 1, value: 1 },
        ];
        add_explicit_geometry(&mut seeded_window, 4, 1);
        let envelope = classify_decode_envelope(&seeded_window).unwrap().unwrap();
        assert_eq!((envelope.token_count, envelope.lane_count), (4, 1));
        assert_eq!(envelope.token_indptr, vec![0, 4]);
        seeded_window.stages[0].ops.push(Op::ChanPut { chan: split, value: 1 });
        assert!(classify_decode_envelope(&seeded_window).is_err());

        let mut multi_lane = section3_container();
        multi_lane.channels[0].shape = Shape::vector(4);
        multi_lane.channels[1].shape = Shape::vector(4);
        multi_lane.ports[1] = const_port(Port::EmbedIndptr, &[0, 1, 2, 3, 4]);
        multi_lane.stages[0].ops = vec![
            Op::ChanPut { chan: 0, value: 0 },
            Op::ChanPut { chan: 1, value: 1 },
        ];
        add_explicit_geometry(&mut multi_lane, 4, 4);
        let envelope = classify_decode_envelope(&multi_lane).unwrap().unwrap();
        assert_eq!((envelope.token_count, envelope.lane_count), (4, 4));
        assert_eq!(
            envelope.template(&multi_lane).unwrap().qo_indptr,
            vec![0, 1, 2, 3, 4]
        );
        let template = envelope.template(&multi_lane).unwrap();
        assert_eq!(template.sampling_indices, vec![0, 0, 0, 0]);
        assert_eq!(template.sampling_indptr, vec![0, 1, 2, 3, 4]);
    }

    /// Rectangular batch of `b` lanes, full KV arity from ports.
    fn beam_container(b: u32, p: u32) -> TraceContainer {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(b), Dtype::I32),    // 0 toks
                chan(Shape::vector(b), Dtype::U32),    // 1 pos
                chan(Shape::matrix(b, p), Dtype::U32), // 2 pages
                chan(Shape::vector(b), Dtype::U32),    // 3 klen
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                const_port(Port::EmbedIndptr, &(0..=b).collect::<Vec<_>>()),
                PortBinding {
                    port: Port::Positions,
                    source: PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::Pages,
                    source: PortSource::Channel(2),
                },
                const_port(
                    Port::PageIndptr,
                    &(0..=b).map(|i| i * p).collect::<Vec<_>>(),
                ),
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(3),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanTake(0)],
            }],
        }
    }

    #[test]
    fn beam_rectangular_batch_geometry() {
        let c = beam_container(2, 3);
        let values: Vec<Option<Vec<u8>>> = vec![
            Some(u32_bytes(&[100, 200])), // 0 toks (reinterpret i32→u32)
            Some(u32_bytes(&[7, 9])),     // 1 pos
            Some(u32_bytes(&[10, 11, 12, 20, 21, 22])), // 2 pages [B,P] flat
            Some(u32_bytes(&[20, 33])),   // 3 klen (physical spans)
        ];
        let g = map_geometry(&c, &values).unwrap();

        assert_eq!(g.token_ids, vec![100, 200]);
        assert_eq!(g.qo_indptr, vec![0, 1, 2], "one token per lane");
        assert_eq!(g.position_ids, vec![7, 9]);
        assert_eq!(
            g.sampling_indices,
            vec![0, 1],
            "last token of each of 2 lanes"
        );
        assert_eq!(g.sampling_indptr, vec![0, 1, 2]);
        assert_eq!(g.kv_page_indices, vec![10, 11, 12, 20, 21, 22]);
        assert_eq!(g.kv_page_indptr, vec![0, 3, 6]);
        // The extents the `kv_len` port stated, undivided: the page size is
        // whoever-owns-the-page-table's business now.
        assert_eq!(g.kv_len, vec![20, 33]);
    }

    /// A device-geometry container's ports are unfilled at host fire time;
    /// the strict map refuses to invent them (the engine resolves them
    /// in-graph and the host maps the result through `map_geometry_evaluated_with`).
    #[test]
    fn unfilled_device_ports_are_rejected() {
        let c = beam_container(2, 3);
        let values: Vec<Option<Vec<u8>>> = vec![None, None, None, None];
        assert!(
            map_geometry(&c, &values).is_err(),
            "strict gate errors on device-resolved ports"
        );
    }

}
