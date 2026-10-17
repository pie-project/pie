use eta_compiler::eval::pareval::GeometryTaint;
use eta_ir::container::{ChanDType, PortSource, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Port, PortMask};
use eta_ir::types::Dtype;
use grammar::brle::RunMask;

#[cfg(test)]
pub fn classify_decode_envelope(
    container: &TraceContainer,
    taint: &GeometryTaint,
) -> Result<Option<PortMask>, String> {
    classify_decode_envelope_why(container, taint, &mut String::new())
}

/// Tokens (and mask, fold and positions) are the device's; where rows land
/// stays the host's.
pub fn classify_decode_envelope_why(
    container: &TraceContainer,
    taint: &GeometryTaint,
    why: &mut String,
) -> Result<Option<PortMask>, String> {
    let mut decline = |reason: String| -> Result<Option<PortMask>, String> {
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
    let device = |port| taint.device_dependent_ports.contains(&port);
    if device(Port::Slots) || device(Port::KvLen) {
        return decline(
            "where its rows land or what they read is computed on the device".to_string(),
        );
    }
    let Some(token_channel) = channel_index(Port::EmbedTokens) else {
        return decline("EmbedTokens is not bound to a channel".to_string());
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
    if !token.seeded && !loop_carried {
        return decline("the token channel is neither seeded nor loop-carried".to_string());
    }
    let token_dims = token.shape.dims();
    if token_dims.len() != 1
        || token_dims[0] == 0
        || !matches!(
            token.dtype,
            ChanDType::Concrete(Dtype::I32) | ChanDType::Concrete(Dtype::U32)
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
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| u32::from_le_bytes(*bytes))
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
                || !matches!(declaration.dtype, ChanDType::Concrete(Dtype::U32))
            {
                return Err(format!(
                    "decode envelope EmbedIndptr channel must be a [lanes + 1] u32 vector with \
                     at most {token_count} lane(s)"
                ));
            }
            if dims[0] == token_count + 1 {
                (0..=token_count).collect()
            } else if declaration.seeded && !puts_channel(*channel as usize) {
                if dims[0] == 2 {
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
    let device_positions = device(Port::Positions);
    if device_positions {
        if channel_for(Port::Slots).is_none() {
            return decline(
                "its positions are computed on the device and it states no slots, so they \
                 are where its rows land"
                    .to_string(),
            );
        }
        let declaration = container
            .channels
            .get(channel_index(Port::Positions).unwrap_or(usize::MAX))
            .ok_or_else(|| "decode envelope position channel is out of range".to_string())?;
        if declaration.shape.dims() != [token_count]
            || !matches!(declaration.dtype, ChanDType::Concrete(Dtype::U32))
        {
            return Err(format!(
                "device-carried positions must be a [{token_count}] u32 vector"
            ));
        }
    }
    let device_fold = device(Port::Fold);
    if device_fold {
        let declaration = container
            .channels
            .get(channel_index(Port::Fold).unwrap_or(usize::MAX))
            .ok_or_else(|| "decode envelope fold channel is out of range".to_string())?;
        let dims = declaration.shape.dims();
        if !(dims == [lane_count] || dims == [1])
            || !matches!(
                declaration.dtype,
                ChanDType::Concrete(Dtype::U32) | ChanDType::Concrete(Dtype::I32)
            )
        {
            return Err(format!(
                "a fold computed on the device must be a [{lane_count}] (or [1]) u32/i32 vector"
            ));
        }
    }
    let device_mask = device(Port::AttnMask);
    if device_mask {
        let declaration = container
            .channels
            .get(channel_index(Port::AttnMask).unwrap_or(usize::MAX))
            .ok_or_else(|| "decode envelope mask channel is out of range".to_string())?;
        if !matches!(declaration.dtype, ChanDType::Concrete(Dtype::Bool)) {
            return Err("device attention mask must be a bool channel".to_string());
        }
        let dims = declaration.shape.dims();
        let rows_per_fire = match dims {
            [rows, keys] if *keys > 0 => *rows,
            [keys] if *keys > 0 => 1,
            _ => {
                return decline(format!(
                    "a device attention mask is a [tokens, keys] (or [keys]) bool channel; \
                     this one is {dims:?}"
                ));
            }
        };
        if rows_per_fire != token_count {
            return decline(format!(
                "the device attention mask holds {rows_per_fire} row(s) and the fire embeds \
                 {token_count} token(s); one row per token"
            ));
        }
    }
    if let Some(port) = taint.device_dependent_ports.iter().find(|port| {
        !matches!(
            port,
            Port::EmbedTokens | Port::Positions | Port::Fold | Port::AttnMask
        )
    }) {
        return decline(format!("{port:?} is computed on the device"));
    }
    let mut required = PortMask::of(&[Port::EmbedTokens]);
    if device_positions {
        required = required.with(Port::Positions);
    }
    if device_fold {
        required = required.with(Port::Fold);
    }
    if device_mask {
        required = required.with(Port::AttnMask);
    }
    Ok(Some(required))
}

/// A pass whose program re-publishes its tokens and where they land in a stage.
pub fn detect_pooled_device_geometry(container: &TraceContainer) -> Option<Vec<u32>> {
    let channel_of = |port: Port| {
        container
            .ports
            .iter()
            .find_map(|binding| match &binding.source {
                PortSource::Channel(channel) if binding.port == port => Some(*channel as usize),
                _ => None,
            })
    };
    let republished = |channel: usize| {
        container.stages.iter().any(|stage| {
            stage
                .ops
                .iter()
                .any(|op| matches!(op, Op::ChanPut { chan, .. } if *chan as usize == channel))
        })
    };

    if let Some(mask) = channel_of(Port::AttnMask) {
        if !matches!(
            container.channels.get(mask)?.dtype,
            ChanDType::Concrete(Dtype::Bool)
        ) {
            return None;
        }
        if !republished(mask) {
            return None;
        }
    }
    // A port the pass's own epilogue re-publishes, or one no host can write
    // and no seed fills (unseeded, host role none): another pass of the
    // same program puts it on the device (a queue between passes).
    let device_fed = |channel: usize| {
        republished(channel)
            || container.channels.get(channel).is_some_and(|declared| {
                declared.host_role == eta_ir::container::HostRole::None && !declared.seeded
            })
    };
    let lands = channel_of(Port::Slots).or_else(|| channel_of(Port::Positions))?;
    if !device_fed(lands) || !device_fed(channel_of(Port::EmbedTokens)?) {
        return None;
    }
    for port in [Port::Positions, Port::Slots, Port::KvLen] {
        if let Some(channel) = channel_of(port)
            && !device_fed(channel)
        {
            return None;
        }
    }

    let tokens = container.channels.get(channel_of(Port::EmbedTokens)?)?;
    let dims = tokens.shape.dims();
    if dims.len() != 1 || dims[0] == 0 {
        return None;
    }
    if !matches!(
        tokens.dtype,
        ChanDType::Concrete(Dtype::I32) | ChanDType::Concrete(Dtype::U32)
    ) {
        return None;
    }
    let count = dims[0];
    match channel_of(Port::EmbedIndptr) {
        Some(split) if !republished(split) => {
            let declaration = container.channels.get(split)?;
            if !declaration.seeded || !matches!(declaration.dtype, ChanDType::Concrete(Dtype::U32))
            {
                return None;
            }
            match declaration.shape.dims() {
                [2] => Some(vec![0, count]),
                [bounds] if *bounds == count + 1 => Some((0..=count).collect()),
                _ => None,
            }
        }
        _ => Some((0..=count).collect()),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReqGeometry {
    pub token_ids: Vec<u32>,
    pub qo_indptr: Vec<u32>,
    pub positions: Option<Vec<u32>>,
    pub slots: Option<Vec<u32>>,
    pub kv_len: Option<Vec<u32>>,
    pub fold: Fold,
    pub sampling_indices: Vec<u32>,
    pub sampling_indptr: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Fold {
    #[default]
    None,
    Host(Vec<u32>),
    Device,
}

impl Fold {
    pub fn of(&self, lane: usize) -> Option<u32> {
        match self {
            Fold::None => Some(0),
            Fold::Host(counts) => counts.get(lane).or(counts.first()).copied(),
            Fold::Device => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub folded: u32,
    pub folded_exact: bool,
    pub tail: Option<u32>,
    pub shared: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Landing {
    pub start: u32,
    pub rows: u32,
}

impl Landing {
    pub fn end(self) -> u32 {
        self.start + self.rows
    }
}

impl ReqGeometry {
    /// Rows land at their slots or after the last slot written, and read every
    /// slot before their last.
    pub fn lanes(
        &self,
        places: &[Place],
        page: u32,
    ) -> Result<(Vec<::engine::Lane>, Vec<Landing>), String> {
        let cut_by = |values: &[u32], indptr: &[u32], lane: usize| -> Vec<u32> {
            let (Some(&start), Some(&end)) = (indptr.get(lane), indptr.get(lane + 1)) else {
                return Vec::new();
            };
            values
                .get(start as usize..end as usize)
                .unwrap_or_default()
                .to_vec()
        };
        let cut = |values: &[u32], lane: usize| cut_by(values, &self.qo_indptr, lane);
        let count = self.qo_indptr.len().saturating_sub(1);
        if places.len() != count {
            return Err(format!(
                "the pass binds {} working set(s) for its {count} lane(s); give one per lane",
                places.len()
            ));
        }
        let page = page.max(1);
        let mut lanes = Vec::with_capacity(count);
        let mut landings = Vec::with_capacity(count);
        for (lane, place) in places.iter().enumerate() {
            let tokens = cut(&self.token_ids, lane);
            let rows = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
            let start = match &self.slots {
                Some(slots) => {
                    let slots = cut(slots, lane);
                    let Some(&first) = slots.first() else {
                        return Err(format!("lane {lane} states no slot for its rows"));
                    };
                    if slots
                        .iter()
                        .enumerate()
                        .any(|(at, &slot)| slot != first + at as u32)
                    {
                        return Err(format!(
                            "lane {lane} writes slots {slots:?}; a lane's rows write \
                             consecutive slots"
                        ));
                    }
                    first
                }
                None if place.shared => {
                    return Err(format!(
                        "lane {lane} shares its working set with another lane and states no \
                         slots; lanes that share a working set state where their rows land"
                    ));
                }
                None => place.tail.ok_or_else(|| {
                    format!(
                        "lane {lane} appends, but where its working set's last write ended was \
                         computed on the device; state its slots"
                    )
                })?,
            };
            if place.folded_exact && start < place.folded {
                return Err(format!(
                    "lane {lane} writes slot {start}, inside the frozen prefix (0..{}); write \
                     at or after it",
                    place.folded
                ));
            }
            let end = start + rows;
            let kv_len = match &self.kv_len {
                Some(lens) => {
                    let len = lens.get(lane).copied().ok_or_else(|| {
                        format!("kv-len states {} length(s) for {count} lane(s)", lens.len())
                    })?;
                    if len != end {
                        return Err(format!(
                            "lane {lane} reads {len} slot(s) but its last row is slot {}; a lane \
                             the host places reads up to its last row",
                            end.saturating_sub(1)
                        ));
                    }
                    len
                }
                None => end,
            };
            let positions = match &self.positions {
                Some(positions) => {
                    let positions = cut(positions, lane);
                    if positions
                        .iter()
                        .enumerate()
                        .all(|(at, &position)| position == start + at as u32)
                    {
                        Vec::new()
                    } else {
                        positions
                    }
                }
                None => Vec::new(),
            };
            let readout = cut_by(&self.sampling_indices, &self.sampling_indptr, lane);
            lanes.push(::engine::Lane {
                tokens,
                positions,
                kv: ::engine::KvDelta {
                    held: start,
                    pages: (0..kv_len.div_ceil(page)).collect(),
                    ..Default::default()
                },
                rs: ::engine::RsVerb::Fold,
                rs_reset: ::engine::RsReset::Inferred,
                readout: match readout.as_slice() {
                    [] => ::engine::Readout::None,
                    [only] if *only + 1 == rows => ::engine::Readout::Last,
                    rows => ::engine::Readout::Rows(rows.to_vec()),
                },
                stream: engine::fire::LaneStream::Text,
                ..::engine::Lane::default()
            });
            landings.push(Landing { start, rows });
        }
        Ok((lanes, landings))
    }
}

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
    pub(crate) fn apply_to(self, request: &mut crate::engine::FireRequest) -> Result<(), String> {
        match self {
            FireAttnMask::Omitted => {}
            FireAttnMask::Host { masks, mask_indptr } => {
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
                        [] => None,
                        [only] => Some(::engine::Masking::Extent(only.clone())),
                        rows => Some(::engine::Masking::Rows(rows.to_vec())),
                    };
                }
                request.has_user_mask = true;
                request.single_token_mode = false;
            }
            FireAttnMask::Device => {
                request.has_user_mask = true;
            }
        }
        Ok(())
    }
}

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

pub type PortEvaluations = Vec<(Port, Result<eta_compiler::eval::interp::Value, String>)>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluatedGeometryError {
    NotDerivable { port: Port, blocker: String },
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

pub fn map_geometry_evaluated_with(
    bound: &eta_ir::validate::BoundTrace,
    known: &mut dyn FnMut(u32) -> Option<eta_compiler::eval::interp::Value>,
    device_resolved: PortMask,
) -> Result<(ReqGeometry, PortEvaluations), EvaluatedGeometryError> {
    use eta_compiler::eval::interp::Value;

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
    let optional_u32 = |port: Port| -> Result<Option<Vec<u32>>, EvaluatedGeometryError> {
        match port_value(port) {
            Some(Ok(value)) => Ok(Some(value_as_u32(&value))),
            Some(Err(blocker)) => Err(EvaluatedGeometryError::NotDerivable { port, blocker }),
            None => Ok(None),
        }
    };
    let required_u32 = |port: Port| -> Result<Vec<u32>, EvaluatedGeometryError> {
        optional_u32(port)?.ok_or_else(|| EvaluatedGeometryError::BadValue {
            port,
            reason: "port is not bound".to_string(),
        })
    };

    let qo_indptr = required_u32(Port::EmbedIndptr)?;
    let spanned = qo_indptr.last().copied().unwrap_or(0) as usize;
    let lanes = qo_indptr.len().saturating_sub(1);
    let token_ids = match required_u32(Port::EmbedTokens) {
        Ok(ids) => ids,
        Err(EvaluatedGeometryError::NotDerivable { .. })
            if device_resolved.contains(Port::EmbedTokens) =>
        {
            vec![0; spanned]
        }
        Err(error) => return Err(error),
    };
    let positions = match optional_u32(Port::Positions) {
        Ok(positions) => positions,
        Err(EvaluatedGeometryError::NotDerivable { .. })
            if device_resolved.contains(Port::Positions) =>
        {
            None
        }
        Err(error) => return Err(error),
    };
    let fold = match optional_u32(Port::Fold) {
        Ok(Some(counts)) => Fold::Host(counts),
        Ok(None) => Fold::None,
        Err(EvaluatedGeometryError::NotDerivable { .. })
            if device_resolved.contains(Port::Fold) =>
        {
            Fold::Device
        }
        Err(error) => return Err(error),
    };
    let mut g = ReqGeometry {
        token_ids,
        qo_indptr,
        positions,
        slots: optional_u32(Port::Slots)?,
        kv_len: optional_u32(Port::KvLen)?,
        fold,
        ..ReqGeometry::default()
    };
    for (port, rows) in [
        (Port::EmbedTokens, Some(&mut g.token_ids)),
        (Port::Positions, g.positions.as_mut()),
        (Port::Slots, g.slots.as_mut()),
    ] {
        let Some(rows) = rows else { continue };
        if spanned == 0 {
            rows.clear();
        } else if rows.len() != spanned {
            return Err(EvaluatedGeometryError::BadValue {
                port,
                reason: format!(
                    "the token CSR spans {spanned} rows but {} were supplied",
                    rows.len()
                ),
            });
        }
    }
    if let Some(kv_len) = &g.kv_len
        && kv_len.len() != lanes
    {
        return Err(EvaluatedGeometryError::BadValue {
            port: Port::KvLen,
            reason: format!(
                "expected one length for each of {lanes} lanes, got {}",
                kv_len.len()
            ),
        });
    }

    let readout = match optional_u32(Port::Readout)? {
        Some(readout) => readout,
        None => g
            .qo_indptr
            .windows(2)
            .filter(|lane| lane[1] > lane[0])
            .map(|lane| lane[1] - 1)
            .collect(),
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

    let evaluated = ports
        .into_iter()
        .map(|(port, slot)| (port, slot.map_err(|blocker| blocker.to_string())))
        .collect();
    Ok((g, evaluated))
}

pub(crate) fn value_as_u32(value: &eta_compiler::eval::interp::Value) -> Vec<u32> {
    use eta_compiler::eval::interp::Value;
    match value {
        Value::U32(v) => v.clone(),
        Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
        Value::F32(v) => v.iter().map(|&x| x as u32).collect(),
        Value::Bool(v) => v.iter().map(|&b| b as u32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_ir::container::{ChannelDecl, HostRole, PortBinding, StageProgram};
    use eta_ir::registry::Stage;
    use eta_ir::types::Shape;
    use std::collections::BTreeSet;

    fn chan(shape: Shape, dtype: Dtype) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    fn decode(lanes: u32, extra: &[(Port, Shape, Dtype)]) -> TraceContainer {
        let mut container = TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![chan(Shape::vector(lanes), Dtype::I32)],
            ports: vec![PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(0),
            }],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanPut { chan: 0, value: 0 }],
            }],
        };
        for (port, shape, dtype) in extra {
            let index = container.channels.len() as u32;
            container.channels.push(chan(*shape, *dtype));
            container.ports.push(PortBinding {
                port: *port,
                source: PortSource::Channel(index),
            });
        }
        container
    }

    fn taint(ports: &[Port]) -> GeometryTaint {
        GeometryTaint {
            device_decided: BTreeSet::new(),
            device_dependent_ports: ports.iter().copied().collect(),
        }
    }

    fn place(tail: u32) -> Place {
        Place {
            folded: 0,
            folded_exact: true,
            tail: Some(tail),
            shared: false,
        }
    }

    fn geometry(rows: &[u32]) -> ReqGeometry {
        let mut qo_indptr = vec![0];
        for &n in rows {
            qo_indptr.push(qo_indptr.last().unwrap() + n);
        }
        let total = *qo_indptr.last().unwrap();
        ReqGeometry {
            token_ids: (0..total).collect(),
            sampling_indices: rows.iter().map(|&n| n - 1).collect(),
            sampling_indptr: (0..=rows.len() as u32).collect(),
            qo_indptr,
            ..ReqGeometry::default()
        }
    }

    fn masked_decode(lanes: u32, pool: u32) -> TraceContainer {
        let decls = [
            (Port::EmbedTokens, Shape::vector(lanes), Dtype::I32),
            (Port::Positions, Shape::vector(lanes), Dtype::U32),
            (Port::KvLen, Shape::vector(lanes), Dtype::U32),
            (Port::AttnMask, Shape::matrix(lanes, pool), Dtype::Bool),
            (Port::Slots, Shape::vector(lanes), Dtype::U32),
        ];
        let mut container = TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
        };
        for (port, shape, dtype) in decls {
            let index = container.channels.len() as u32;
            container.channels.push(chan(shape, dtype));
            container.ports.push(PortBinding {
                port,
                source: PortSource::Channel(index),
            });
            container.stages[0].ops.push(Op::ChanPut {
                chan: index,
                value: 0,
            });
        }
        container
    }

    fn channel_of(container: &TraceContainer, port: Port) -> u32 {
        container
            .ports
            .iter()
            .find_map(|binding| match (&binding.port, &binding.source) {
                (bound, PortSource::Channel(channel)) if *bound == port => Some(*channel),
                _ => None,
            })
            .expect("port bound")
    }

    #[test]
    fn geometry_every_case() {
        a_decode_that_states_nothing_rides_the_envelope();
        a_device_mask_and_fold_ride_the_envelope();
        slots_computed_on_the_device_leave_the_envelope();
        device_positions_ride_only_beside_stated_slots();
        a_lane_appends_after_its_tail();
        stated_slots_land_where_they_say();
        a_write_inside_the_frozen_prefix_is_refused();
        lanes_sharing_a_working_set_state_their_slots();
        a_host_lane_reads_up_to_its_last_row();
        masked_loop_carried_decode_is_pooled_device_geometry();
        a_mask_free_decode_that_republishes_every_port_is_pooled_too();
        a_decode_whose_slots_the_host_states_keeps_the_envelope_path();
        a_decode_stating_no_slots_is_not_pooled();
        a_decode_another_pass_lays_out_is_pooled_device_geometry();
    }

    fn a_decode_another_pass_lays_out_is_pooled_device_geometry() {
        // A block drafter's loop: the verify lays out the draft's geometry,
        // so the draft's ports are unseeded, host-blind and never put by
        // the draft itself.
        let mut container = masked_decode(1, 128);
        container
            .ports
            .retain(|binding| binding.port != Port::AttnMask);
        container.stages[0].ops.clear();
        for declared in &mut container.channels {
            declared.seeded = false;
        }
        assert_eq!(
            detect_pooled_device_geometry(&container),
            Some(vec![0, 1]),
            "a port another pass of the program puts is device-resolvable"
        );
        container.channels[0].host_role = HostRole::Writer;
        assert_eq!(
            detect_pooled_device_geometry(&container),
            None,
            "a port the host writes is not"
        );
    }

    fn a_decode_that_states_nothing_rides_the_envelope() {
        assert_eq!(
            classify_decode_envelope(&decode(1, &[]), &taint(&[Port::EmbedTokens])),
            Ok(Some(PortMask::of(&[Port::EmbedTokens]))),
            "an appending decode is the envelope's"
        );
    }

    fn a_device_mask_and_fold_ride_the_envelope() {
        let container = decode(
            1,
            &[
                (Port::AttnMask, Shape::matrix(1, 64), Dtype::Bool),
                (Port::Fold, Shape::vector(1), Dtype::U32),
            ],
        );
        assert_eq!(
            classify_decode_envelope(
                &container,
                &taint(&[Port::EmbedTokens, Port::AttnMask, Port::Fold]),
            ),
            Ok(Some(PortMask::of(&[
                Port::EmbedTokens,
                Port::AttnMask,
                Port::Fold
            ]))),
            "mask and fold may come from the device"
        );
    }

    fn slots_computed_on_the_device_leave_the_envelope() {
        let container = decode(2, &[(Port::Slots, Shape::vector(2), Dtype::U32)]);
        assert_eq!(
            classify_decode_envelope(&container, &taint(&[Port::EmbedTokens, Port::Slots])),
            Ok(None)
        );
    }

    fn device_positions_ride_only_beside_stated_slots() {
        let alone = decode(1, &[(Port::Positions, Shape::vector(1), Dtype::U32)]);
        let device = taint(&[Port::EmbedTokens, Port::Positions]);
        assert_eq!(classify_decode_envelope(&alone, &device), Ok(None));
        let beside = decode(
            1,
            &[
                (Port::Positions, Shape::vector(1), Dtype::U32),
                (Port::Slots, Shape::vector(1), Dtype::U32),
            ],
        );
        assert_eq!(
            classify_decode_envelope(&beside, &device),
            Ok(Some(PortMask::of(&[Port::EmbedTokens, Port::Positions]))),
            "RoPE from the device, slots from the host"
        );
    }

    fn a_lane_appends_after_its_tail() {
        let (lanes, landings) = geometry(&[3]).lanes(&[place(40)], 16).unwrap();
        assert_eq!(landings, vec![Landing { start: 40, rows: 3 }]);
        assert_eq!(lanes[0].kv.held, 40);
        assert_eq!(
            lanes[0].kv.pages,
            vec![0, 1, 2],
            "slots 0..43 span three pages"
        );
        assert!(lanes[0].positions.is_empty(), "positions follow the slots");
        assert_eq!(lanes[0].readout, ::engine::Readout::Last);
    }

    fn stated_slots_land_where_they_say() {
        let mut g = geometry(&[2, 1]);
        g.slots = Some(vec![5, 6, 9]);
        g.positions = Some(vec![5, 6, 2]);
        let (lanes, landings) = g.lanes(&[place(7), place(0)], 4).unwrap();
        assert_eq!(
            landings,
            vec![Landing { start: 5, rows: 2 }, Landing { start: 9, rows: 1 }]
        );
        assert!(lanes[0].positions.is_empty());
        assert_eq!(
            lanes[1].positions,
            vec![2],
            "a RoPE position apart from its slot"
        );
        g.slots = Some(vec![5, 7, 9]);
        assert!(
            g.lanes(&[place(7), place(0)], 4)
                .unwrap_err()
                .contains("consecutive")
        );
    }

    fn a_write_inside_the_frozen_prefix_is_refused() {
        let mut g = geometry(&[1]);
        g.slots = Some(vec![3]);
        let frozen = Place {
            folded: 4,
            ..place(8)
        };
        assert!(
            g.lanes(&[frozen], 16)
                .unwrap_err()
                .contains("frozen prefix")
        );
        let unsure = Place {
            folded_exact: false,
            ..frozen
        };
        assert!(
            g.lanes(&[unsure], 16).is_ok(),
            "past a fold on the device the host cannot tell"
        );
    }

    fn lanes_sharing_a_working_set_state_their_slots() {
        let shared = Place {
            shared: true,
            ..place(0)
        };
        assert!(
            geometry(&[1, 1])
                .lanes(&[shared, shared], 16)
                .unwrap_err()
                .contains("state where their rows land")
        );
    }

    fn a_host_lane_reads_up_to_its_last_row() {
        let mut g = geometry(&[2]);
        g.kv_len = Some(vec![12]);
        assert!(g.lanes(&[place(10)], 16).is_ok());
        g.kv_len = Some(vec![20]);
        assert!(g.lanes(&[place(10)], 16).unwrap_err().contains("last row"));
    }

    fn masked_loop_carried_decode_is_pooled_device_geometry() {
        assert_eq!(
            detect_pooled_device_geometry(&masked_decode(1, 128)),
            Some(vec![0, 1])
        );
        assert_eq!(
            detect_pooled_device_geometry(&masked_decode(2, 128)),
            Some(vec![0, 1, 2])
        );
    }

    fn a_mask_free_decode_that_republishes_every_port_is_pooled_too() {
        let mut container = masked_decode(2, 128);
        container
            .ports
            .retain(|binding| binding.port != Port::AttnMask);
        assert_eq!(
            detect_pooled_device_geometry(&container),
            Some(vec![0, 1, 2])
        );
    }

    fn a_decode_whose_slots_the_host_states_keeps_the_envelope_path() {
        let mut container = masked_decode(1, 128);
        let slots = channel_of(&container, Port::Slots);
        container.stages[0]
            .ops
            .retain(|op| !matches!(op, Op::ChanPut { chan, .. } if *chan == slots));
        assert_eq!(
            detect_pooled_device_geometry(&container),
            None,
            "slots the program never re-publishes are the host's"
        );
    }

    fn a_decode_stating_no_slots_is_not_pooled() {
        let mut container = masked_decode(1, 128);
        container
            .ports
            .retain(|binding| !matches!(binding.port, Port::Slots | Port::Positions));
        assert_eq!(
            detect_pooled_device_geometry(&container),
            None,
            "a decode that appends is the host's to place"
        );
    }
}
