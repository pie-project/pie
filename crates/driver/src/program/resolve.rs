use driver_api::program::{LaunchPackage, LaunchPort};
use tensor_ir::registry::Port;

use super::channel::InterpInstance;
use super::plan::{ConstPortValue, ExecPlan};
use super::value::Value;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    pub token_ids: Vec<u32>,

    pub qo_indptr: Vec<u32>,

    pub position_ids: Vec<u32>,

    pub kv_page_indptr: Vec<u32>,

    pub kv_page_indices: Vec<u32>,

    pub kv_last_page_lens: Vec<u32>,

    pub has_kv_family: bool,

    pub sampling_indices: Vec<u32>,

    pub sampling_indptr: Vec<u32>,

    pub w_page: Vec<u32>,

    pub w_off: Vec<u32>,

    pub has_write_desc: bool,

    pub mask: Vec<u8>,

    pub has_mask: bool,

    pub mask_key_len: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    Ready(Box<Geometry>),

    NotReady { channel: u32 },

    Failed { message: String },
}

fn as_u32(value: &Value) -> Option<Vec<u32>> {
    match value {
        Value::U32(lanes) => Some(lanes.clone()),
        Value::I32(lanes) => Some(lanes.iter().map(|&lane| lane as u32).collect()),
        _ => None,
    }
}

enum Source<'a> {
    Channel(u32),

    Const(&'a Value),
}

const PORTS: usize = Port::ALL.len();

struct Bindings<'a> {
    sources: [Option<Source<'a>>; PORTS],
}

impl<'a> Bindings<'a> {
    fn of(package: &'a LaunchPackage, constants: &'a [ConstPortValue]) -> Result<Self, Resolution> {
        let mut sources: [Option<Source<'a>>; PORTS] = [const { None }; PORTS];
        for binding in &package.ports {
            // No `Port::from_u8` guard any more: `LaunchPort::port` is a
            // `Port`, so a binding naming no port is not a binding.
            let port = binding.port;
            sources[port as usize] = Some(if binding.is_const {
                Source::Const(constant_for(binding, constants)?)
            } else {
                Source::Channel(binding.channel)
            });
        }
        Ok(Self { sources })
    }

    fn get(&self, port: Port) -> Option<&Source<'a>> {
        self.sources[port as usize].as_ref()
    }

    fn bound(&self, port: Port) -> bool {
        self.sources[port as usize].is_some()
    }
}

fn constant_for<'a>(
    binding: &LaunchPort,
    constants: &'a [ConstPortValue],
) -> Result<&'a Value, Resolution> {
    constants
        .iter()
        .find(|folded| folded.port == binding.port)
        .map(|folded| &folded.value)
        .ok_or_else(|| Resolution::Failed {
            message: format!(
                "port {} is declared const but no folded value was kept for it",
                binding.port.name()
            ),
        })
}

fn read_u32(
    bindings: &Bindings<'_>,
    instance: &InterpInstance,
    port: Port,
) -> Result<Vec<u32>, Resolution> {
    match bindings.get(port) {
        Some(Source::Channel(channel)) => {
            let ring =
                instance
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| Resolution::Failed {
                        message: format!(
                            "port {port:?} names channel {channel}, which is not bound"
                        ),
                    })?;
            if ring.is_empty() {
                return Err(Resolution::NotReady { channel: *channel });
            }
            as_u32(&ring.front()).ok_or_else(|| Resolution::Failed {
                message: format!(
                    "the cell on channel {channel} for port {port:?} is not an integer type, \
                     so it cannot be a geometry index"
                ),
            })
        }
        Some(Source::Const(value)) => as_u32(value).ok_or_else(|| Resolution::Failed {
            message: format!("the folded constant for port {port:?} is not an integer type"),
        }),
        None => Ok(Vec::new()),
    }
}

fn read_mask(bindings: &Bindings<'_>, instance: &InterpInstance) -> Result<Vec<u8>, Resolution> {
    let port = Port::AttnMask;
    let value = match bindings.get(port) {
        Some(Source::Channel(channel)) => {
            let ring =
                instance
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| Resolution::Failed {
                        message: format!(
                            "the mask port names channel {channel}, which is not bound"
                        ),
                    })?;
            if ring.is_empty() {
                return Err(Resolution::NotReady { channel: *channel });
            }
            ring.front()
        }
        Some(Source::Const(value)) => (*value).clone(),
        None => return Ok(Vec::new()),
    };
    match value {
        Value::Bool(lanes) => Ok(lanes),
        other => Err(Resolution::Failed {
            message: format!(
                "the attention mask cell is {:?}, not bool; a dense mask is one byte per lane",
                other.dtype()
            ),
        }),
    }
}

#[must_use]
pub const fn last_page_len(len: u32, page: u32) -> u32 {
    if page == 0 || len == 0 {
        return 0;
    }
    ((len - 1) % page) + 1
}

#[must_use]
pub fn resolve(plan: &ExecPlan, instance: &InterpInstance, page: u32) -> Resolution {
    match resolve_inner(plan, instance, page) {
        Ok(geometry) => Resolution::Ready(Box::new(geometry)),
        Err(resolution) => resolution,
    }
}

#[allow(clippy::too_many_lines)]
fn resolve_inner(
    plan: &ExecPlan,
    instance: &InterpInstance,
    page: u32,
) -> Result<Geometry, Resolution> {
    let bindings = Bindings::of(&plan.package, &plan.const_ports)?;

    let mut out = Geometry {
        token_ids: read_u32(&bindings, instance, Port::EmbedTokens)?,
        ..Geometry::default()
    };
    let tokens = u32::try_from(out.token_ids.len()).unwrap_or(u32::MAX);

    out.qo_indptr = if bindings.bound(Port::EmbedIndptr) {
        read_u32(&bindings, instance, Port::EmbedIndptr)?
    } else {
        vec![0, tokens]
    };
    let requests = out.qo_indptr.len().saturating_sub(1);

    out.position_ids = if bindings.bound(Port::Positions) {
        read_u32(&bindings, instance, Port::Positions)?
    } else {
        (0..tokens).collect()
    };

    out.kv_page_indptr = read_u32(&bindings, instance, Port::PageIndptr)?;
    if bindings.bound(Port::Pages) {
        out.kv_page_indices = read_u32(&bindings, instance, Port::Pages)?;

        if let Some(&live) = out.kv_page_indptr.last() {
            let live = live as usize;
            if live <= out.kv_page_indices.len() {
                out.kv_page_indices.truncate(live);
            }
        }
        out.has_kv_family = true;
    }
    if bindings.bound(Port::KvLen) {
        out.kv_last_page_lens = read_u32(&bindings, instance, Port::KvLen)?
            .into_iter()
            .map(|len| last_page_len(len, page))
            .collect();
    }

    out.sampling_indptr.push(0);
    if bindings.bound(Port::Readout) {
        out.sampling_indices = read_u32(&bindings, instance, Port::Readout)?;
        if requests <= 1 {
            out.sampling_indptr
                .push(u32::try_from(out.sampling_indices.len()).unwrap_or(u32::MAX));
        } else if out.sampling_indices.len() == requests {
            for lane in 1..=requests {
                out.sampling_indptr
                    .push(u32::try_from(lane).unwrap_or(u32::MAX));
            }
        } else {
            return Err(Resolution::Failed {
                message: format!(
                    "the readout port carries {} index(es) for {requests} requests; a \
                     multi-request readout needs exactly one per request",
                    out.sampling_indices.len()
                ),
            });
        }
    } else {
        for lane in 0..requests {
            let span = out.qo_indptr[lane + 1].saturating_sub(out.qo_indptr[lane]);
            if span != 0 {
                out.sampling_indices.push(span - 1);
            }
            out.sampling_indptr
                .push(u32::try_from(out.sampling_indices.len()).unwrap_or(u32::MAX));
        }
    }

    if bindings.bound(Port::WSlot) {
        out.w_page = read_u32(&bindings, instance, Port::WSlot)?;
        out.has_write_desc = true;
    }
    if bindings.bound(Port::WOff) {
        out.w_off = read_u32(&bindings, instance, Port::WOff)?;
    }

    if bindings.bound(Port::AttnMask) {
        out.mask = read_mask(&bindings, instance)?;
        out.has_mask = true;

        if !out.token_ids.is_empty() && out.mask.len().is_multiple_of(out.token_ids.len()) {
            out.mask_key_len =
                u32::try_from(out.mask.len() / out.token_ids.len()).unwrap_or(u32::MAX);
        }
    }

    Ok(out)
}
