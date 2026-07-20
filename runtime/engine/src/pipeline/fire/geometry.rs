//! Descriptor-ports → `LaunchPlan` geometry mapping (thrust-3 P2c-fire).
//!
//! A **pure function** from a trace container's descriptor ports (+ the current
//! per-channel values at fire time) to the request's forward geometry: the
//! token family (`embed_tokens`/`positions`/`embed_indptr`/`readout` →
//! `token_ids`/`position_ids`/`qo_indptr`/`sampling_*`) and the port-provided KV
//! family (`pages`/`page_indptr`/`kv_len` → `kv_page_indices`/`kv_page_indptr`/
//! `kv_last_page_lens`). Unit-testable in isolation against the locked
//! bound program contract — no driver decode, no GPU.
//!
//! The one geometry piece that is NOT pure-from-ports is the **sugar** KV arity
//! (`attn_working_set(&ws, &len)`, only `kv_len` bound): the page indices come
//! from the instance's `WorkingSet` allocation, not a port — that derivation is
//! staged for the fire wiring (needs the ws page map). This mapper fills every
//! field the ports DO provide; absent-port fields stay empty for the caller.
//!
//! Complete pipeline domain API: some methods here (relaxed geometry
//! variants, per-channel introspection, the pure `instantiate`/registry
//! probe entry points, device-geometry lease internals) are not yet
//! called by the current single-model/mock-driver fire path, but are
//! exercised by this module's own unit tests and reserved for upcoming
//! wiring (multi-pass channels, device-geometry beams) — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use pie_grammar::brle::RunMask;
use pie_ptir::container::{PortSource, TraceContainer};
use pie_ptir::op::Op;
use pie_ptir::registry::Port;
use pie_ptir::types::DType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeEnvelope {
    pub token_count: u32,
    pub lane_count: u32,
    pub token_indptr: Vec<u32>,
    pub loop_carried: bool,
    /// `Positions` binds a channel (device-carried) rather than a const —
    /// executing the class demands the positions device port.
    pub device_positions: bool,
    /// `AttnMask` binds a channel (dense device-carried bool mask) —
    /// executing the class demands the mask descriptor port, and the fire
    /// is marked mask-carrying so the scheduler keeps it solo.
    pub has_mask: bool,
}

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
        let readout = match const_port(container, Port::Readout) {
            Some(bytes) => as_u32(Port::Readout, bytes)?,
            None => qo_indptr
                .windows(2)
                .map(|lane| lane[1].saturating_sub(1))
                .collect(),
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
            ..ReqGeometry::default()
        })
    }
}

/// Pure shape classification of the decode-envelope family. Capability is
/// the CALLER's decision: a shape match on a driver without the device
/// geometry ports falls back to host-evaluated (serialized) execution
/// rather than erroring — derivability decides class, the driver's port
/// mask only decides where the class executes.
pub fn classify_decode_envelope(
    container: &TraceContainer,
) -> Result<Option<DecodeEnvelope>, String> {
    if !container.externs.is_empty() {
        return Ok(None);
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
        return Ok(None);
    };
    let Some(kv_len_channel) = channel_index(Port::KvLen) else {
        return Ok(None);
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
        return Ok(None);
    }
    for port in [
        Port::Positions,
        Port::Pages,
        Port::PageIndptr,
        Port::WSlot,
        Port::WOff,
    ] {
        if channel_for(port).is_none() {
            return Ok(None);
        }
    }
    let token_dims = token.shape.dims();
    if token_dims.len() != 1
        || token_dims[0] == 0
        || !matches!(
            token.dtype,
            pie_ptir::container::ChanDType::Concrete(DType::I32)
                | pie_ptir::container::ChanDType::Concrete(DType::U32)
        )
    {
        return Err("decode envelope tokens must be a non-empty i32/u32 vector".to_string());
    }
    let token_count = token_dims[0];
    let qo_indptr = match channel_for(Port::EmbedIndptr) {
        None => vec![0, token_count],
        Some(PortSource::Const { dtype, shape, data })
            if *dtype == DType::U32 && shape.dims().len() == 1 =>
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
            if declaration.shape.dims() != [token_count + 1]
                || !matches!(
                    declaration.dtype,
                    pie_ptir::container::ChanDType::Concrete(DType::U32)
                )
            {
                return Err(format!(
                    "decode envelope EmbedIndptr channel must be a [{}] u32 vector",
                    token_count + 1
                ));
            }
            (0..=token_count).collect()
        }
        Some(_) => {
            return Err("decode envelope EmbedIndptr must be a u32 vector".to_string());
        }
    };
    if qo_indptr.len() < 2
        || qo_indptr[0] != 0
        || qo_indptr.last().copied() != Some(token_count)
        || qo_indptr.windows(2).any(|pair| pair[1] != pair[0] + 1)
    {
        return Err("decode envelope EmbedIndptr must declare one token per lane".to_string());
    }
    let lane_count = (qo_indptr.len() - 1) as u32;
    if kv_len.shape.dims() != [lane_count]
        || !matches!(
            kv_len.dtype,
            pie_ptir::container::ChanDType::Concrete(DType::U32)
        )
    {
        return Err(format!(
            "decode envelope KV length must be a [{lane_count}] u32 vector"
        ));
    }

    let mut device_positions = false;
    let mut has_mask = false;
    for binding in &container.ports {
        match (&binding.port, &binding.source) {
            (Port::EmbedTokens | Port::KvLen, PortSource::Channel(_)) => {}
            (Port::EmbedIndptr, PortSource::Const { dtype, shape, data })
                if *dtype == DType::U32
                    && shape.dims() == [lane_count + 1]
                    && data.len() == (lane_count as usize + 1) * 4 => {}
            (Port::EmbedIndptr, PortSource::Channel(channel)) => {
                let declaration = container.channels.get(*channel as usize).ok_or_else(|| {
                    "decode envelope EmbedIndptr channel is out of range".to_string()
                })?;
                if declaration.shape.dims() != [lane_count + 1]
                    || !matches!(
                        declaration.dtype,
                        pie_ptir::container::ChanDType::Concrete(DType::U32)
                    )
                {
                    return Err("device EmbedIndptr must be a [lanes+1] u32 vector".to_string());
                }
            }
            (Port::Positions, PortSource::Const { dtype, shape, .. })
                if *dtype == DType::U32 && shape.dims() == [token_count] => {}
            (Port::Readout, PortSource::Const { dtype, shape, data })
                if *dtype == DType::U32
                    && shape.dims().len() == 1
                    && data.len() == shape.dims()[0] as usize * 4
                    && data.chunks_exact(4).all(|bytes| {
                        u32::from_le_bytes(bytes.try_into().unwrap()) < token_count
                    }) => {}
            (Port::PageIndptr, PortSource::Const { dtype, shape, data })
                if *dtype == DType::U32
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
                        pie_ptir::container::ChanDType::Concrete(DType::U32)
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
                if declaration.shape.dims().len() != 2
                    || declaration.shape.dims()[0] != lane_count
                    || declaration.shape.dims()[1] == 0
                    || !matches!(
                        declaration.dtype,
                        pie_ptir::container::ChanDType::Concrete(DType::U32)
                    )
                {
                    return Err(
                        "device pages must be a non-empty [lanes,pages] u32 matrix".to_string()
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
                        pie_ptir::container::ChanDType::Concrete(DType::U32)
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
                        pie_ptir::container::ChanDType::Concrete(DType::U32)
                    )
                {
                    return Err("device WSlot/WOff must be a [tokens] u32 vector".to_string());
                }
            }
            (Port::AttnMask, PortSource::Channel(channel)) => {
                // A dense device-carried mask: the driver resolves the bool
                // cells pre-forward (sink/sliding-window decode). The shape's
                // key extent evolves with the KV, so only the element type is
                // checked here; the driver derives key_len from the cell.
                has_mask = true;
                let declaration = container
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| "decode envelope mask channel is out of range".to_string())?;
                if !matches!(
                    declaration.dtype,
                    pie_ptir::container::ChanDType::Concrete(DType::Bool)
                ) {
                    return Err("device attention mask must be a bool channel".to_string());
                }
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
        has_mask,
    }))
}

/// The device geometry ports executing `envelope` as the DecodeEnvelope
/// class demands of a driver.
pub fn envelope_required_ports(envelope: &DecodeEnvelope) -> u32 {
    let mut required =
        pie_driver_abi::PIE_DEVICE_PORT_EMBED_TOKENS | pie_driver_abi::PIE_DEVICE_PORT_KV_LEN;
    if envelope.device_positions {
        required |= pie_driver_abi::PIE_DEVICE_PORT_POSITIONS;
    }
    if envelope.has_mask {
        required |= pie_driver_abi::PIE_DEVICE_PORT_ATTN_MASK;
    }
    required
}

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

/// The forward geometry a PTIR pass contributes to a `LaunchPlan`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReqGeometry {
    /// Input token ids (from `embed_tokens`).
    pub token_ids: Vec<u32>,
    /// RoPE positions (from `positions`, else append-order `0..nnz`).
    pub position_ids: Vec<u32>,
    /// Per-lane token CSR (from `embed_indptr`, else one lane over all tokens).
    pub qo_indptr: Vec<u32>,
    /// KV page slot ids (from `pages`; empty for the sugar arity — ws-derived).
    pub kv_page_indices: Vec<u32>,
    /// Per-lane page CSR (from `page_indptr`; empty for the sugar arity).
    pub kv_page_indptr: Vec<u32>,
    /// Valid tokens in each lane's last KV page (derived from `kv_len`).
    pub kv_last_page_lens: Vec<u32>,
    /// Read-out positions (from `readout`, else the last token of each lane).
    pub sampling_indices: Vec<u32>,
    /// Per-lane read-out CSR.
    pub sampling_indptr: Vec<u32>,
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
    /// No `embed_tokens` port — every pass embeds tokens (§5.1).
    NoEmbed,
}

impl ReqGeometry {
    /// Write this geometry into a [`LaunchPlan`]'s forward-geometry fields
    /// (the host-known prefill for a PTIR fire). One-to-one with the request's
    /// `token_ids`/`position_ids`/`qo_indptr`/`kv_page_*`/`kv_last_page_lens`/
    /// `sampling_*` — leaves every other field (samplers, masks, carrier) intact.
    pub fn apply_to(&self, req: &mut crate::driver::LaunchPlan) {
        req.token_ids = self.token_ids.clone();
        req.position_ids = self.position_ids.clone();
        req.qo_indptr = self.qo_indptr.clone();
        req.kv_page_indices = self.kv_page_indices.clone();
        req.kv_page_indptr = self.kv_page_indptr.clone();
        req.kv_last_page_lens = self.kv_last_page_lens.clone();
        req.sampling_indices = self.sampling_indices.clone();
        req.sampling_indptr = self.sampling_indptr.clone();
    }
}

/// Per-fire lowering of the optional attention-mask descriptor.
///
/// A channel-backed mask is not intrinsically device-resident: a seed or
/// host-staged value is available through the host shadow on this fire, while a
/// value written from a device-only epilogue becomes unknown on a later fire.
/// Keep that distinction per fire so host-known masks use the ordinary wire
/// BRLE path and only genuinely device-derived values select dense device
/// lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FireAttnMask {
    Omitted,
    Host {
        masks: Vec<crate::driver::command::EncodedMask>,
        mask_indptr: Vec<u32>,
    },
    Device,
}

impl FireAttnMask {
    pub(crate) fn apply_to(self, request: &mut crate::driver::LaunchPlan) {
        match self {
            FireAttnMask::Omitted => {}
            FireAttnMask::Host { masks, mask_indptr } => {
                request.masks = masks;
                request.mask_indptr = mask_indptr;
                request.has_user_mask = true;
                // A decode-shaped custom mask still needs the mask-aware
                // prefill attention path.
                request.single_token_mode = false;
            }
            FireAttnMask::Device => {
                request.has_user_mask = true;
            }
        }
    }
}

/// Lower an already-evaluated `AttnMask` port into one BRLE row per query.
pub(crate) fn lower_attn_mask_evaluated(
    container: &TraceContainer,
    qo_indptr: &[u32],
    evaluated: &[(Port, Result<pie_ptir::interp::Value, String>)],
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
    let pie_ptir::interp::Value::Bool(dense) = value else {
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
            crate::driver::command::EncodedMask::new(mask.buffer, mask.total_size)
        })
        .collect();
    Ok(FireAttnMask::Host {
        masks,
        mask_indptr: qo_indptr.to_vec(),
    })
}

/// Evaluate and lower the mask against this fire's host-shadow value oracle.
pub(crate) fn evaluate_attn_mask(
    bound: &pie_ptir::validate::BoundTrace,
    known: &mut dyn FnMut(u32) -> Option<pie_ptir::interp::Value>,
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
    let evaluated = pie_ptir::pareval::eval_descriptor_ports(bound, known)
        .map_err(|blocker| format!("attention-mask evaluation failed: {blocker}"))?
        .into_iter()
        .map(|(port, value)| (port, value.map_err(|blocker| blocker.to_string())))
        .collect::<Vec<_>>();
    lower_attn_mask_evaluated(&bound.container, qo_indptr, &evaluated)
}

/// Per-channel values at fire time: `values[i]` is channel `i`'s current cell
/// bytes (little-endian, per its dtype), or `None` if unfilled.
pub type ChannelValues<'a> = &'a [Option<Vec<u8>>];

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

/// Map a pass's descriptor ports to forward geometry by **evaluating** the
/// geometry prologue over host-known channel values (`pie_ptir::pareval`) —
/// the general form of [`map_geometry`], which reads only directly-present
/// values and is its degenerate case. Returns the geometry plus every port's
/// evaluated value (the canonical-KV gate verifies evidence against these).
///
/// A rank-2 `Pages` envelope (`[lanes, P]`, the SDK lowering) is compacted to
/// the wire CSR by each lane's live page count from `PageIndptr`, mirroring
/// the driver's descriptor resolution; rank-1 pages pass through flat.
pub fn map_geometry_evaluated(
    bound: &pie_ptir::validate::BoundTrace,
    known: &mut dyn FnMut(u32) -> Option<pie_ptir::interp::Value>,
    page_size: u32,
) -> Result<
    (
        ReqGeometry,
        Vec<(Port, Result<pie_ptir::interp::Value, String>)>,
    ),
    EvaluatedGeometryError,
> {
    use pie_ptir::interp::Value;

    let container = &bound.container;
    let ports = pie_ptir::pareval::eval_descriptor_ports(bound, known).map_err(|blocker| {
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

    let mut g = ReqGeometry::default();
    g.token_ids = required_u32(Port::EmbedTokens)?;
    let nnz = g.token_ids.len() as u32;

    g.qo_indptr = optional_u32(Port::EmbedIndptr)?.unwrap_or_else(|| vec![0, nnz]);

    let kv_len = required_u32(Port::KvLen)?;
    let lanes = g.qo_indptr.len().saturating_sub(1);
    let (default_positions, default_pages, default_page_indptr) =
        dense_defaults(&g.qo_indptr, &kv_len, nnz, page_size).map_err(|reason| {
            EvaluatedGeometryError::BadValue {
                port: Port::KvLen,
                reason,
            }
        })?;
    g.position_ids = optional_u32(Port::Positions)?.unwrap_or(default_positions);

    // Read-out rows distribute over lanes as LANE-RELATIVE indices (the
    // multi-row wire contract; identical to the envelope template). Absent
    // readout samples each lane's last row.
    let readout = match optional_u32(Port::Readout)? {
        Some(readout) => readout,
        None => g
            .qo_indptr
            .windows(2)
            .map(|lane| lane[1].saturating_sub(1))
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

    let pages = optional_u32(Port::Pages)?;
    let page_indptr = optional_u32(Port::PageIndptr)?;
    match (pages, page_indptr) {
        (Some(pages), Some(indptr)) => {
            g.kv_page_indices =
                compact_page_envelope(container, pages, &indptr).map_err(|reason| {
                    EvaluatedGeometryError::BadValue {
                        port: Port::Pages,
                        reason,
                    }
                })?;
            g.kv_page_indptr = indptr;
        }
        (Some(pages), None) => {
            g.kv_page_indices = compact_page_envelope(container, pages, &default_page_indptr)
                .map_err(|reason| EvaluatedGeometryError::BadValue {
                    port: Port::Pages,
                    reason,
                })?;
            g.kv_page_indptr = default_page_indptr;
        }
        (None, Some(indptr)) => {
            g.kv_page_indices = default_pages;
            g.kv_page_indptr = indptr;
        }
        (None, None) => {
            g.kv_page_indices = default_pages;
            g.kv_page_indptr = default_page_indptr;
        }
    }
    if kv_len.len() != lanes {
        return Err(EvaluatedGeometryError::BadValue {
            port: Port::KvLen,
            reason: format!(
                "expected one length for each of {lanes} lanes, got {}",
                kv_len.len()
            ),
        });
    }
    g.kv_last_page_lens = kv_len
        .iter()
        .map(|&len| last_page_len(len, page_size))
        .collect();

    let evaluated = ports
        .into_iter()
        .map(|(port, slot)| (port, slot.map_err(|blocker| blocker.to_string())))
        .collect();
    Ok((g, evaluated))
}

/// Reinterpret an evaluated value's lanes as `u32` (i32 tokens bit-cast, the
/// driver's `token_ids` convention; bool as 0/1).
pub(crate) fn value_as_u32(value: &pie_ptir::interp::Value) -> Vec<u32> {
    use pie_ptir::interp::Value;
    match value {
        Value::U32(v) => v.clone(),
        Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
        Value::F32(v) => v.iter().map(|&x| x as u32).collect(),
        Value::Bool(v) => v.iter().map(|&b| b as u32).collect(),
    }
}

/// Compact a `Pages` port value to the wire lane-page CSR: a rank-2
/// `[lanes, P]` envelope (the SDK lowering) keeps each lane's live prefix per
/// `page_indptr`'s counts, mirroring the driver's descriptor resolution;
/// rank-1 pages are already flat and pass through.
pub(crate) fn compact_page_envelope(
    container: &TraceContainer,
    pages: Vec<u32>,
    page_indptr: &[u32],
) -> Result<Vec<u32>, String> {
    let dims = port_dims(container, Port::Pages).unwrap_or_default();
    if dims.len() != 2 {
        return Ok(pages);
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

/// Map a container's ports to the forward geometry (P2c-fire, pure).
pub fn map_geometry(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    page_size: u32,
) -> Result<ReqGeometry, GeometryError> {
    map_geometry_impl(container, values, page_size, false)
}

/// **Relaxed** geometry map for a DEVICE-GEOMETRY fire (plan W3.4): a descriptor
/// port bound to a device-produced channel that has no host-known value at fire
/// time leaves its wire field EMPTY instead of erroring
/// ([`GeometryError::MissingChannelValue`]). The driver resolves those ports
/// pre-forward from the channel cells (W1.1); the host still prefills whatever
/// const / host-known ports it can. `NoEmbed` is likewise relaxed (empty
/// tokens). `BadPayload` still errors — a malformed const is a real bug.
pub fn map_geometry_relaxed(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    page_size: u32,
) -> Result<ReqGeometry, GeometryError> {
    map_geometry_impl(container, values, page_size, true)
}

fn map_geometry_impl(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    page_size: u32,
    relaxed: bool,
) -> Result<ReqGeometry, GeometryError> {
    let mut g = ReqGeometry::default();

    // -- token family --
    // In relaxed mode a missing embed channel leaves tokens empty (the driver
    // reads them pre-forward); strict mode errors as before.
    let tokens = match resolve_opt(container, values, Port::EmbedTokens, relaxed)? {
        Some(t) => t,
        None if relaxed => Vec::new(),
        None => return Err(GeometryError::NoEmbed),
    };
    g.token_ids = as_u32(Port::EmbedTokens, &tokens)?;
    let nnz = g.token_ids.len() as u32;

    g.qo_indptr = match resolve_opt(container, values, Port::EmbedIndptr, relaxed)? {
        Some(b) => as_u32(Port::EmbedIndptr, &b)?,
        None => vec![0, nnz], // one lane over all tokens
    };
    let lanes = g.qo_indptr.len().saturating_sub(1);

    let kv_len = match resolve_opt(container, values, Port::KvLen, relaxed)? {
        Some(b) => Some(as_u32(Port::KvLen, &b)?),
        None if relaxed => None,
        None => return Err(GeometryError::BadCsr { port: Port::KvLen }),
    };
    let defaults = kv_len
        .as_deref()
        .map(|lengths| dense_defaults(&g.qo_indptr, lengths, nnz, page_size))
        .transpose()
        .map_err(|_| GeometryError::BadCsr { port: Port::KvLen })?;
    g.position_ids = match resolve_opt(container, values, Port::Positions, relaxed)? {
        Some(b) => as_u32(Port::Positions, &b)?,
        None => defaults
            .as_ref()
            .map(|(positions, _, _)| positions.clone())
            .unwrap_or_default(),
    };

    // read-out: explicit positions, else the last token of each lane.
    match resolve_opt(container, values, Port::Readout, relaxed)? {
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

    let explicit_pages = resolve_opt(container, values, Port::Pages, relaxed)?
        .map(|b| as_u32(Port::Pages, &b))
        .transpose()?;
    let explicit_indptr = resolve_opt(container, values, Port::PageIndptr, relaxed)?
        .map(|b| as_u32(Port::PageIndptr, &b))
        .transpose()?;
    match (explicit_pages, explicit_indptr, defaults.as_ref()) {
        (Some(pages), Some(indptr), _) => {
            g.kv_page_indices = compact_page_envelope(container, pages, &indptr)
                .map_err(|_| GeometryError::BadCsr { port: Port::Pages })?;
            g.kv_page_indptr = indptr;
        }
        (Some(pages), None, Some((_, _, indptr))) => {
            g.kv_page_indices = compact_page_envelope(container, pages, indptr)
                .map_err(|_| GeometryError::BadCsr { port: Port::Pages })?;
            g.kv_page_indptr = indptr.to_vec();
        }
        (None, Some(indptr), Some((_, pages, _))) => {
            g.kv_page_indices = pages.clone();
            g.kv_page_indptr = indptr;
        }
        (None, None, Some((_, pages, indptr))) => {
            g.kv_page_indices = pages.clone();
            g.kv_page_indptr = indptr.clone();
        }
        (Some(pages), None, None) => g.kv_page_indices = pages,
        (None, Some(indptr), None) => g.kv_page_indptr = indptr,
        (None, None, None) => {}
    }
    if let Some(kv_len) = kv_len {
        g.kv_last_page_lens = kv_len
            .iter()
            .map(|&len| last_page_len(len, page_size))
            .collect();
    }

    Ok(g)
}

/// Valid tokens in the last KV page for a physical span `len` (§5.1): a full
/// span ends the page exactly (`page_size`), else the remainder; `0` for empty.
fn last_page_len(len: u32, page_size: u32) -> u32 {
    if len == 0 || page_size == 0 {
        0
    } else {
        ((len - 1) % page_size) + 1
    }
}

fn dense_defaults(
    qo_indptr: &[u32],
    kv_len: &[u32],
    token_count: u32,
    page_size: u32,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), String> {
    if page_size == 0
        || qo_indptr.len() != kv_len.len() + 1
        || qo_indptr.first().copied() != Some(0)
        || qo_indptr.last().copied() != Some(token_count)
        || qo_indptr.windows(2).any(|lane| lane[1] < lane[0])
        || qo_indptr
            .windows(2)
            .zip(kv_len)
            .any(|(lane, &len)| lane[1] - lane[0] > len)
    {
        return Err("dense geometry has inconsistent lane metadata".to_string());
    }
    const MAX_DENSE_PAGE_REFS: u64 = 1_048_576;
    let page_refs = kv_len.iter().try_fold(0u64, |total, &len| {
        total.checked_add(u64::from(len.div_ceil(page_size)))
    });
    if page_refs.is_none_or(|page_refs| page_refs > MAX_DENSE_PAGE_REFS) {
        return Err("dense geometry exceeds the page-reference safety bound".to_string());
    }

    let mut positions = Vec::with_capacity(token_count as usize);
    let mut pages = Vec::with_capacity(page_refs.unwrap() as usize);
    let mut page_indptr = Vec::with_capacity(qo_indptr.len());
    page_indptr.push(0);
    for (lane, &len) in qo_indptr.windows(2).zip(kv_len) {
        let query_len = lane[1] - lane[0];
        let start = len
            .checked_sub(query_len)
            .ok_or_else(|| "KV length is shorter than its query lane".to_string())?;
        positions.extend(start..len);
        pages.extend(0..len.div_ceil(page_size));
        page_indptr.push(
            u32::try_from(pages.len()).map_err(|_| "dense page CSR exceeds u32".to_string())?,
        );
    }
    Ok((positions, pages, page_indptr))
}

/// Resolve a port's value: its const payload, or the current value of the
/// channel it binds. `None` if the container has no such port.
fn resolve(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    port: Port,
) -> Result<Option<Vec<u8>>, GeometryError> {
    resolve_opt(container, values, port, false)
}

/// Like [`resolve`] but, in `relaxed` mode, a port bound to a channel with no
/// host-known value returns `None` (the driver resolves it pre-forward, W1.1)
/// instead of erroring [`GeometryError::MissingChannelValue`].
fn resolve_opt(
    container: &TraceContainer,
    values: ChannelValues<'_>,
    port: Port,
    relaxed: bool,
) -> Result<Option<Vec<u8>>, GeometryError> {
    let Some(binding) = container.ports.iter().find(|p| p.port == port) else {
        return Ok(None);
    };
    match &binding.source {
        PortSource::Const { data, .. } => Ok(Some(data.clone())),
        PortSource::Channel(c) => match values.get(*c as usize).and_then(|v| v.clone()) {
            Some(v) => Ok(Some(v)),
            None if relaxed => Ok(None),
            None => Err(GeometryError::MissingChannelValue { port, channel: *c }),
        },
    }
}

/// Reinterpret a little-endian byte payload as `u32`s (4 bytes each). Token ids
/// stored `i32` reinterpret bit-for-bit (the driver's `token_ids` is `u32`).
fn as_u32(port: Port, bytes: &[u8]) -> Result<Vec<u32>, GeometryError> {
    if bytes.len() % 4 != 0 {
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
    use pie_ptir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::op::Op;
    use pie_ptir::registry::Stage;
    use pie_ptir::types::{DType, Shape};

    fn u32_bytes(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
    fn const_port(port: Port, words: &[u32]) -> PortBinding {
        PortBinding {
            port,
            source: PortSource::Const {
                dtype: DType::U32,
                shape: Shape::vector(words.len() as u32),
                data: u32_bytes(words),
            },
        }
    }
    fn chan(shape: Shape, dtype: DType) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    /// §3 sugar: embed tok (chan 0) + embed_indptr const [0,1] + kv_len (chan 1).
    fn section3_container() -> TraceContainer {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32), // 0 tok
                chan(Shape::vector(1), DType::U32), // 1 len
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
            .push(chan(Shape::vector(tokens), DType::U32));
        let pages = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::matrix(lanes, 2), DType::U32));
        let page_indptr = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::vector(lanes + 1), DType::U32));
        let w_slot = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::vector(tokens), DType::U32));
        let w_off = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::vector(tokens), DType::U32));
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
        let c = section3_container();
        // tok = [42] (i32), len = [5] (u32); page_size 16.
        let values: Vec<Option<Vec<u8>>> = vec![
            Some(42i32.to_le_bytes().to_vec()),
            Some(5u32.to_le_bytes().to_vec()),
        ];
        let g = map_geometry(&c, &values, 16).unwrap();

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
            g.kv_last_page_lens,
            vec![5],
            "len 5 in a 16-page → last page holds 5"
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
        container.channels.push(chan(Shape::vector(2), DType::U32));
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
    fn decode_envelope_accepts_a_device_carried_bool_mask() {
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
            .push(chan(Shape::matrix(1, 8), DType::Bool));
        container.ports.push(PortBinding {
            port: Port::AttnMask,
            source: PortSource::Channel(mask),
        });

        let envelope = classify_decode_envelope(&container)
            .unwrap()
            .expect("masked loop-carried decode classifies as an envelope");
        assert!(envelope.has_mask);
        assert_ne!(
            envelope_required_ports(&envelope) & pie_driver_abi::PIE_DEVICE_PORT_ATTN_MASK,
            0,
            "a masked envelope demands the mask descriptor port"
        );
        // RV-26: the geometry ports alone must NOT satisfy a masked
        // envelope. A driver that advertises geometry ports but cannot
        // execute per-lane masks in its envelope compose (CUDA today)
        // routes masked envelopes through the loud Host fallback — a
        // driver that CLAIMS the mask port turns that fallback into a
        // bind error, which is why claiming it demands real execution.
        let required = envelope_required_ports(&envelope);
        assert_ne!(
            required & pie_driver_abi::PIE_DEVICE_GEOMETRY_PORTS,
            required,
            "geometry ports alone must not admit a masked envelope"
        );

        // A non-bool mask channel is a classification error, not a fallback.
        let bad = container.channels.len() as u32 - 1;
        container.channels[bad as usize].dtype = ChanDType::Concrete(DType::U32);
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
        assert!(classify_decode_envelope(&multi_token).is_err());

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

    /// §6.2 rectangular batch: B=2 lanes, full KV arity from ports.
    fn beam_container(b: u32, p: u32) -> TraceContainer {
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(b), DType::I32),    // 0 toks
                chan(Shape::vector(b), DType::U32),    // 1 pos
                chan(Shape::matrix(b, p), DType::U32), // 2 pages
                chan(Shape::vector(b), DType::U32),    // 3 klen
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
        let g = map_geometry(&c, &values, 16).unwrap();

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
        // len 20 in 16-page → last page holds 4; len 33 → 1.
        assert_eq!(g.kv_last_page_lens, vec![4, 1]);
    }

    #[test]
    fn missing_channel_value_errors() {
        let c = section3_container();
        let values: Vec<Option<Vec<u8>>> = vec![None, Some(5u32.to_le_bytes().to_vec())];
        let e = map_geometry(&c, &values, 16).unwrap_err();
        assert_eq!(
            e,
            GeometryError::MissingChannelValue {
                port: Port::EmbedTokens,
                channel: 0
            }
        );
    }

    /// W3.4 relaxed gate: a device-geometry fire whose port channels are unfilled
    /// leaves the wire fields EMPTY (the driver resolves them pre-forward, W1.1)
    /// instead of erroring — but const/host-known ports still prefill.
    #[test]
    fn relaxed_leaves_device_ports_empty() {
        let c = beam_container(2, 3);
        // Nothing produced yet (all device channels unfilled).
        let values: Vec<Option<Vec<u8>>> = vec![None, None, None, None];
        let g = map_geometry_relaxed(&c, &values, 16).unwrap();
        assert!(g.token_ids.is_empty(), "device embed_tokens left empty");
        assert!(g.kv_page_indices.is_empty(), "device pages left empty");
        assert!(g.kv_last_page_lens.is_empty(), "device kv_len left empty");
        // The const embed_indptr port ([0,1,2]) still prefills.
        assert_eq!(
            g.qo_indptr,
            vec![0, 1, 2],
            "const port prefilled even in relaxed mode"
        );
        // strict mode still errors on the same input.
        assert!(
            map_geometry(&c, &values, 16).is_err(),
            "strict gate still errors"
        );
    }

    #[test]
    fn last_page_len_boundaries() {
        assert_eq!(last_page_len(0, 16), 0);
        assert_eq!(last_page_len(1, 16), 1);
        assert_eq!(last_page_len(16, 16), 16, "a full page ends exactly");
        assert_eq!(last_page_len(17, 16), 1);
        assert_eq!(last_page_len(32, 16), 16);
    }

    #[test]
    fn dense_defaults_reject_unbounded_page_materialization() {
        assert!(
            dense_defaults(&[0, 1], &[u32::MAX], 1, 16).is_err(),
            "guest KvLen must not force an unbounded host allocation"
        );
    }

    #[test]
    fn dense_defaults_reject_malformed_query_csr_before_allocating() {
        assert!(dense_defaults(&[0, u32::MAX], &[1], 1, 16).is_err());
    }

    fn mask_container() -> TraceContainer {
        let mut container = section3_container();
        let mask = container.channels.len() as u32;
        container
            .channels
            .push(chan(Shape::matrix(2, 4), DType::Bool));
        container.ports.push(PortBinding {
            port: Port::AttnMask,
            source: PortSource::Channel(mask),
        });
        container
    }

    fn expand_mask(mask: &crate::driver::command::EncodedMask) -> Vec<bool> {
        let mut values = Vec::new();
        for (run, &len) in mask.runs.iter().enumerate() {
            values.extend(std::iter::repeat_n(run % 2 == 1, len as usize));
        }
        values
    }

    #[test]
    fn omitted_attention_mask_stays_mask_free() {
        let lowered = lower_attn_mask_evaluated(&section3_container(), &[0, 1], &[]).unwrap();
        assert_eq!(lowered, FireAttnMask::Omitted);
        let mut plan = crate::driver::LaunchPlan {
            single_token_mode: true,
            ..Default::default()
        };
        lowered.apply_to(&mut plan);
        assert!(!plan.has_user_mask);
        assert!(plan.masks.is_empty());
        assert!(plan.mask_indptr.is_empty());
        assert!(plan.single_token_mode);
    }

    #[test]
    fn host_derived_attention_mask_lowers_to_wire_brle() {
        let dense = vec![true, true, false, true, false, true, true, false];
        let evaluated = vec![(
            Port::AttnMask,
            Ok(pie_ptir::interp::Value::Bool(dense.clone())),
        )];
        let lowered = lower_attn_mask_evaluated(&mask_container(), &[0, 1, 2], &evaluated).unwrap();
        let FireAttnMask::Host { masks, mask_indptr } = &lowered else {
            panic!("host-known mask must use wire lowering");
        };
        assert_eq!(mask_indptr, &[0, 1, 2]);
        assert_eq!(masks.len(), 2);
        assert_eq!(expand_mask(&masks[0]), dense[..4]);
        assert_eq!(expand_mask(&masks[1]), dense[4..]);

        let mut plan = crate::driver::LaunchPlan {
            single_token_mode: true,
            ..Default::default()
        };
        lowered.apply_to(&mut plan);
        assert!(plan.has_user_mask);
        assert_eq!(plan.mask_indptr, vec![0, 1, 2]);
        assert!(
            !plan.single_token_mode,
            "decode-shaped custom masks require the prefill fallback"
        );
    }

    #[test]
    fn attention_mask_classification_is_per_fire() {
        let container = mask_container();
        let host = vec![(
            Port::AttnMask,
            Ok(pie_ptir::interp::Value::Bool(vec![true; 8])),
        )];
        assert!(matches!(
            lower_attn_mask_evaluated(&container, &[0, 1, 2], &host).unwrap(),
            FireAttnMask::Host { .. }
        ));

        let device = vec![(Port::AttnMask, Err("device epilogue put".to_string()))];
        assert_eq!(
            lower_attn_mask_evaluated(&container, &[0, 1, 2], &device).unwrap(),
            FireAttnMask::Device
        );
        let mut plan = crate::driver::LaunchPlan::default();
        FireAttnMask::Device.apply_to(&mut plan);
        assert!(plan.has_user_mask);
        assert!(plan.masks.is_empty(), "dense device path has no wire rows");
    }
}
