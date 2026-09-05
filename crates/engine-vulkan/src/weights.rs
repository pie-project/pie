use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve;
use checkpoint::file::zt;
use checkpoint::plan::{LoadPlan, StorageTarget, compile, compile_streaming};
use checkpoint::serving::Stamp;
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use engine::load::{Residency, Tiers};
use kernels_vulkan::Tensor;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::alloc::Memory;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::experts::{
    Gathered, HostPlane, HostTier, Layout, Mapping, Plan, Pump, Seats, Tier, identity_runs,
};
use crate::run::{WeightRow, WeightTable};
use crate::weight_store::Store;

pub(crate) const ALIGN: u64 = 256;

#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    pub bank: &'a str,

    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
struct Bank {
    offset: u64,

    adapters: u32,

    slot: u64,
    rows: u64,
    cols: u64,
    elem: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSeat {
    pub name: String,
    pub adapters: u32,
    pub slot: u64,

    pub rows: u64,

    pub cols: u64,
    pub elem: u64,
}

fn banks(trace: &Trace, places: &[Place]) -> BTreeMap<String, Bank> {
    trace
        .params
        .iter()
        .zip(places)
        .filter(|(param, _)| param.source == ParamSource::Registered)
        .map(|(param, place)| {
            let adapters =
                u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(u32::MAX);
            let slot = match adapters {
                0 => 0,
                seats => place.bytes / u64::from(seats),
            };

            let (rows, cols) = rectangle(param.shape.get(1..).unwrap_or(&[]));
            (
                param.name.clone(),
                Bank {
                    offset: place.offset,
                    adapters,
                    slot,
                    rows,
                    cols,
                    elem: model_compiler::arena::elem_bytes(param.dtype).unwrap_or(0),
                },
            )
        })
        .collect()
}

#[derive(Debug)]
pub struct Weights {
    store: Store,

    host: Option<HostTier>,

    layout: Layout,

    seats: Option<Seats>,

    pump: Option<Pump>,

    rows: Option<Gathered>,
    table: WeightTable,

    banks: BTreeMap<String, Bank>,

    decoded: Vec<Buffer>,
}

impl Weights {
    #[allow(clippy::too_many_arguments)]
    pub fn resident(
        device: &Context,
        handles: &Handles,
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        device_cap: u64,
        residency: Residency,
        max_tokens: u32,
    ) -> Result<Weights> {
        readable_plane_orders(trace)?;

        serves_this_deployment(path, trace.platform.backend(), &trace.name)?;

        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };

        let target = StorageTarget::for_backend(BackendKind::Vulkan, 0, 1);
        let landing = compile(&metadata, contract, target.clone())?;

        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();

        let pairings = pairings(&landing, &index)?;

        let reserved: Vec<u64> = plane_bytes(trace)?
            .iter()
            .map(|bytes| bytes.next_multiple_of(ALIGN))
            .collect();
        let mut planes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (name, pairing) in &pairings {
            let at = index[name];
            let mut rows = vec![pairing.scales];
            rows.extend(pairing.biases);
            planes.insert(at, rows);
        }
        let plan = Plan::of(trace)?;

        let pinned: Vec<bool> = trace
            .params
            .iter()
            .map(|param| param.source != ParamSource::Checkpoint)
            .collect();
        let layout = plan.tiers(&reserved, &planes, &pinned, device_cap, residency)?;
        let places = places(trace, &layout)?;

        let spans = |tier: Tier| -> Vec<(u64, u64)> {
            places
                .iter()
                .filter(|p| p.tier == tier)
                .map(|p| (p.offset, p.reserved))
                .collect()
        };
        let mut store = Store::with(
            device,
            &spans(Tier::Device),
            device.max_buffer(),
            Memory::Device,
        )?;

        let streaming = compile_streaming(&metadata, contract, target)?;

        let host_map = if layout.streams() {
            if path.is_dir() {
                return Err(Fault::Residency(
                    "a snapshot directory cannot stream routed experts; import it to a \
                     `.zt` artifact first"
                        .into(),
                ));
            }
            Some(Mapping::open(path)?)
        } else {
            None
        };
        let mut host_planes: BTreeMap<usize, HostPlane> = BTreeMap::new();
        if host_map.is_some() {
            for (at, place) in places.iter().enumerate() {
                if place.tier != Tier::Host {
                    continue;
                }
                let name = trace.params[at].name.as_str();
                let runs = identity_runs(&streaming, name, place.bytes).map_err(|why| {
                    Fault::Residency(format!("`{name}` cannot stream from the artifact: {why}"))
                })?;
                let per = crate::experts::row_bytes(place.dtype, place.width);
                host_planes.insert(at, HostPlane::new(runs, place.bytes, per, place.rows));
            }
        }

        let mut sink = Landing {
            store: &mut store,
            host: host_map.as_ref().map(|map| (map, &host_planes)),
            places: &places,
            index: &index,
            landed: vec![false; places.len()],
        };
        Execution::new(&streaming, snapshot)
            .streaming()
            .sink(&mut sink)
            .run()?;
        let landed = sink.landed;

        let mut phantom: BTreeMap<usize, u32> = BTreeMap::new();
        let host = match host_map {
            Some(map) => {
                let order: Vec<usize> = host_planes.keys().copied().collect();
                let planes: Vec<HostPlane> = host_planes.into_values().collect();
                let (tier, minted) = HostTier::new(device, handles, map, planes)?;
                phantom.extend(order.into_iter().zip(minted));
                Some(tier)
            }
            None => None,
        };

        let mut weight_table = Vec::with_capacity(places.len());
        for (at, place) in places.iter().enumerate() {
            if !landed[at] && trace.params[at].source == ParamSource::Checkpoint {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            let dense = |place: &Place, at: usize| -> Result<Tensor> {
                let handle = match place.tier {
                    Tier::Device => store.bind(handles, place.offset, place.bytes)?,
                    Tier::Host => *phantom.get(&at).ok_or(Fault::Ceiling {
                        what: "bytes of the host tier",
                        need: place.bytes,
                        have: 0,
                    })?,
                };
                Ok(Tensor::new(handle, place.rows, place.width, place.dtype))
            };
            weight_table.push(Some(match pairings.get(trace.params[at].name.as_str()) {
                Some(pairing) => WeightRow::Planes(kernels_vulkan::Bank {
                    codes: dense(place, at)?,
                    scales: dense(&places[pairing.scales], pairing.scales)?,
                    biases: pairing.biases.map(|b| dense(&places[b], b)).transpose()?,
                    group: pairing.group,
                    bits: pairing.bits,
                }),
                None => WeightRow::Dense(dense(place, at)?),
            }));
        }

        let mut widest = [0u64; 4];
        let mut capacity = 0u32;
        for &at in plan.routed() {
            if places[at].tier != Tier::Host {
                continue;
            }
            let row = |place: &Place| crate::experts::row_bytes(place.dtype, place.width);
            capacity = capacity.max(places[at].rows);
            match pairings.get(trace.params[at].name.as_str()) {
                Some(pairing) => {
                    widest[0] = widest[0].max(row(&places[at]));
                    widest[1] = widest[1].max(row(&places[pairing.scales]));
                    if let Some(b) = pairing.biases {
                        widest[2] = widest[2].max(row(&places[b]));
                    }
                }
                None => widest[3] = widest[3].max(row(&places[at])),
            }
        }
        let seats = Seats::reserve(device, handles, widest, capacity)?;

        let pump = Pump::reserve(device, handles, layout.slot, layout.ring)?;

        let mut gathered_widest = [0u64; 4];
        for at in plan.gathered() {
            if places[at].tier != Tier::Host {
                continue;
            }
            let row = |place: &Place| crate::experts::row_bytes(place.dtype, place.width);
            match pairings.get(trace.params[at].name.as_str()) {
                Some(pairing) => {
                    gathered_widest[0] = gathered_widest[0].max(row(&places[at]));
                    gathered_widest[1] = gathered_widest[1].max(row(&places[pairing.scales]));
                    if let Some(b) = pairing.biases {
                        gathered_widest[2] = gathered_widest[2].max(row(&places[b]));
                    }
                }

                None => {
                    return Err(Fault::Residency(format!(
                        "`{}` is a dense gathered table on the host tier, and only a \
                         quantized one seats: import it quantized, or raise the device \
                         weight budget so it stays resident",
                        trace.params[at].name
                    )));
                }
            }
        }
        let rows = Gathered::reserve(
            device,
            handles,
            gathered_widest,
            plan.gathered_demand(max_tokens),
        )?;
        Ok(Weights {
            store,
            host,
            layout,
            seats,
            rows,
            pump,
            table: WeightTable(weight_table),
            banks: banks(trace, &places),
            decoded: Vec::new(),
        })
    }

    #[must_use]
    pub fn gathered(&self) -> Option<&Gathered> {
        self.rows.as_ref()
    }

    #[must_use]
    pub fn seats(&self) -> Option<&Seats> {
        self.seats.as_ref()
    }

    #[must_use]
    pub fn pump(&self) -> Option<&Pump> {
        self.pump.as_ref()
    }

    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        for plane in planes {
            let bank = self.banks.get(plane.bank).ok_or_else(|| Fault::Adapter {
                bank: plane.bank.to_string(),
                why: "not a bank this plan declares; a bank is a weight the model text \
                      marked `registered`, and this plan marked none by that name"
                    .to_string(),
            })?;
            if id >= bank.adapters {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} adapters while this registration is id {id}; capacity \
                         is a shape the model text declared, so the fix is the model \
                         text and not a retry",
                        bank.adapters
                    ),
                });
            }
            if plane.bytes.len() as u64 != bank.slot {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} bytes per adapter while this plane carries {}; a plane \
                         is one whole slot, zero-padded by the caller past its own rank",
                        bank.slot,
                        plane.bytes.len()
                    ),
                });
            }
        }
        for plane in planes {
            let bank = self.banks[plane.bank];
            let at = bank.offset + u64::from(id) * bank.slot;
            self.store.zero_span(at, bank.slot)?;
            self.store.write(at, plane.bytes)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.banks
            .iter()
            .map(|(name, bank)| (name.as_str(), bank.adapters, bank.slot))
            .collect()
    }

    #[must_use]
    pub fn bank_seats(&self) -> Vec<BankSeat> {
        self.banks
            .iter()
            .map(|(name, bank)| BankSeat {
                name: name.clone(),
                adapters: bank.adapters,
                slot: bank.slot,
                rows: bank.rows,
                cols: bank.cols,
                elem: bank.elem,
            })
            .collect()
    }

    #[must_use]
    pub fn adapter_seats(&self) -> u32 {
        self.banks
            .values()
            .map(|bank| bank.adapters)
            .min()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn tiers(&self) -> Tiers {
        self.layout.tiers
    }

    #[must_use]
    pub fn streams(&self) -> bool {
        self.layout.streams()
    }

    #[must_use]
    pub fn table(&self) -> &WeightTable {
        &self.table
    }

    pub fn decode_absorbed(
        &mut self,
        device: &Context,
        handles: &Handles,
        trace: &Trace,
    ) -> Result<()> {
        for at in crate::decoded::absorbed_weights(trace) {
            let Some(Some(WeightRow::Planes(bank))) = self.table.0.get(at).copied() else {
                continue;
            };
            let param = &trace.params[at];
            let (n, k) = match param.shape.as_slice() {
                [n, k] => (*n as usize, *k as usize),
                _ => {
                    return Err(Fault::Param {
                        name: param.name.clone(),
                        why: "is not a two-axis plane, the only shape decoded at load",
                    });
                }
            };
            let bits = bank.bits as usize;
            let group = bank.group as usize;
            let codes = handles.read(bank.codes.buf, (n * k * bits / 8) as u64)?;
            let scales = handles.read(bank.scales.buf, (n * k / group * 2) as u64)?;
            let biases = bank
                .biases
                .map(|b| handles.read(b.buf, (n * k / group * 2) as u64))
                .transpose()?;
            let plane = crate::decoded::decode_affine(
                &codes,
                &scales,
                biases.as_deref(),
                n,
                k,
                group,
                bits,
            )
            .map_err(|why| Fault::Device {
                call: "decode_absorbed",
                why: format!("{}: {why}", param.name),
            })?;
            let mut buffer = Buffer::zeroed(device, plane.len() as u64)?;
            buffer.write(0, &plane)?;
            let handle = handles.bind(&buffer, 0, plane.len() as u64)?;
            self.table.0[at] = Some(WeightRow::Dense(Tensor::new(
                handle,
                u32::try_from(n).unwrap_or(u32::MAX),
                u32::try_from(k).unwrap_or(u32::MAX),
                Dtype::Bf16,
            )));
            self.decoded.push(buffer);
        }
        Ok(())
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
            + self.decoded.iter().map(Buffer::bytes).sum::<u64>()
            + self.seats.as_ref().map_or(0, Seats::device_bytes)
            + self.pump.as_ref().map_or(0, Pump::device_bytes)
            + self.rows.as_ref().map_or(0, Gathered::device_bytes)
    }

    #[must_use]
    pub fn host_bytes(&self) -> u64 {
        self.host.as_ref().map_or(0, HostTier::bytes)
    }

    #[must_use]
    pub fn host(&self) -> Option<&HostTier> {
        self.host.as_ref()
    }
}

#[derive(Debug, Clone, Copy)]
struct Pairing {
    scales: usize,

    biases: Option<usize>,
    group: u32,
    bits: u32,
}

fn pairings<'a>(
    landing: &'a LoadPlan,
    index: &BTreeMap<&str, usize>,
) -> Result<BTreeMap<&'a str, Pairing>> {
    let named: BTreeMap<u32, &str> = landing
        .tensors
        .iter()
        .map(|decl| (decl.id.0, decl.name.as_str()))
        .collect();
    let mut out = BTreeMap::new();
    for attachment in &landing.attachments {
        let Some(of) = named.get(&attachment.tensor.0) else {
            continue;
        };

        if !index.contains_key(of) {
            continue;
        }
        let name = of;
        let row = |id: TensorId, what: &'static str| -> Result<usize> {
            named
                .get(&id.0)
                .and_then(|plane| index.get(plane))
                .copied()
                .ok_or_else(|| Fault::Param {
                    name: (*name).to_string(),
                    why: what,
                })
        };
        let bits = match attachment.scale_form {
            ScaleForm::RawE8M0 => 4,
            ScaleForm::Bf16AffineFactors => match landing.affine_point_of(name) {
                Some((_, bits)) => bits,
                None => {
                    return Err(Fault::Param {
                        name: (*name).to_string(),
                        why: "carries affine scale factors and no quantized encoding for \
                              them to be factors OF; the point a kernel is selected at is \
                              the tensor's own `QuantSpec`, and this one has none",
                    });
                }
            },

            ScaleForm::F32Factors => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "wants its scales expanded to f32 factors, and every quantized \
                          point this shell stamps reads them in the width they are stored",
                });
            }
        };

        let biases = match (attachment.scale_form, attachment.zero_point_tensor) {
            (_, Some(id)) => Some(row(
                id,
                "is an affine bank whose zero points this plan does not publish as a \
                 param of their own",
            )?),
            (ScaleForm::Bf16AffineFactors, None) => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "is an affine bank whose scales are half of its dequantization, \
                          and this plan names no zero points for the other half; a \
                          contract states them with `TensorContract::offsetting`",
                });
            }
            (_, None) => None,
        };
        out.insert(
            *name,
            Pairing {
                scales: row(
                    attachment.scale_tensor,
                    "is a quantized weight whose scales this plan does not publish as a \
                     param of their own",
                )?,
                biases,
                group: attachment.group_size,
                bits,
            },
        );
    }
    Ok(out)
}

pub(crate) fn readable_plane_orders(trace: &Trace) -> Result<()> {
    match trace
        .params
        .iter()
        .find(|param| param.dtype == Dtype::U4g64tiled)
    {
        None => Ok(()),
        Some(param) => Err(Fault::Param {
            name: param.name.clone(),
            why: "is declared U4g64tiled — MLX affine codes in m16n8k16 fragment order, which \
                  this shell has no reader for: its qmm and qmv arms index an affine bank \
                  row-major and would answer nonsense off a relaid plane. The order is \
                  `kernels_cuda::linear::tiled`'s, and a model text reaches it only by asking \
                  for it: `model_dsl::place` resolves a placed dtype against the platform the \
                  declaration is read for, and this platform's answer is the canonical \
                  row-major sibling. So either this plane came out of an artifact converted \
                  FOR the cuda shell — convert it again on this box, or serve it there — or a \
                  text stated the order outright, in which case it is the text that has to ask",
        }),
    }
}

pub(crate) fn serves_this_deployment(path: &Path, backend: &str, sku: &str) -> Result<()> {
    if path.is_dir() {
        return Ok(());
    }
    let artifact = match serve::stamp_of(path) {
        Ok(None) => return Ok(()),
        Ok(Some(stamp)) => stamp,

        Err(why) => return Err(Fault::Recipe(format!("checkpoint: {why}"))),
    };
    let deployment = Stamp::of(backend, sku);
    artifact
        .check(&deployment)
        .map_err(|mismatch| Fault::Recipe(mismatch.refuse(&path.display().to_string())))
}

#[derive(Debug, Clone, Copy)]
struct Place {
    tier: Tier,
    offset: u64,

    bytes: u64,

    reserved: u64,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

pub(crate) fn plane_bytes(trace: &Trace) -> Result<Vec<u64>> {
    trace
        .params
        .iter()
        .map(|param| {
            let (rows, width) = rectangle(&param.shape);
            Ok(match param.dtype {
                Dtype::Mxfp4 => rows.saturating_mul(width),

                Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => {
                    rows.saturating_mul(width).div_ceil(2)
                }

                Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => {
                    rows.saturating_mul(width).div_ceil(4)
                }

                Dtype::U8g64 => rows.saturating_mul(width),
                other => {
                    let element =
                        model_compiler::arena::elem_bytes(other).ok_or_else(|| Fault::Param {
                            name: param.name.clone(),
                            why: "is declared in a packed storage element that has no element \
                                  size",
                        })?;
                    rows.saturating_mul(width).saturating_mul(element)
                }
            })
        })
        .collect()
}

fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

fn places(trace: &Trace, layout: &Layout) -> Result<Vec<Place>> {
    let bytes = plane_bytes(trace)?;
    let mut out = Vec::with_capacity(trace.params.len());
    let mut at = [0u64; 2];
    for (index, param) in trace.params.iter().enumerate() {
        let (rows, width) = rectangle(&param.shape);
        let full = bytes[index];
        let tier = layout.tier[index];
        let cursor = &mut at[usize::from(tier == Tier::Host)];
        out.push(Place {
            tier,
            offset: *cursor,
            bytes: full,
            reserved: full.next_multiple_of(ALIGN),
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype: param.dtype,
        });
        *cursor += full.next_multiple_of(ALIGN);
    }
    Ok(out)
}

struct Landing<'a> {
    store: &'a mut Store,

    host: Option<(&'a Mapping, &'a BTreeMap<usize, HostPlane>)>,
    places: &'a [Place],
    index: &'a BTreeMap<&'a str, usize>,
    landed: Vec<bool>,
}

impl TensorSink for Landing<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoadError> {
        let at = *self.index.get(name).ok_or_else(|| {
            LoadError::Contract(format!(
                "the load contract publishes `{name}`, which this plan does not \
                 name — the two were not written from each other"
            ))
        })?;
        let place = self.places[at];
        if bytes.len() as u64 != place.bytes {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {} — a plane read \
                 at the wrong width is a model that computes",
                bytes.len(),
                place.bytes
            )));
        }
        match place.tier {
            Tier::Device => self
                .store
                .write(place.offset, bytes)
                .map_err(|fault| LoadError::Internal(fault.to_string()))?,
            Tier::Host => {
                let (map, planes) = self.host.ok_or_else(|| {
                    LoadError::Internal(format!(
                        "`{name}` is planned for a host tier this load did not map"
                    ))
                })?;
                let plane = planes.get(&at).ok_or_else(|| {
                    LoadError::Internal(format!("`{name}` has no host-tier plane"))
                })?;

                if !plane.agrees(map, bytes) {
                    return Err(LoadError::Contract(format!(
                        "`{name}` as published differs from the artifact's own bytes, \
                         so it cannot stream from the file"
                    )));
                }
            }
        }
        self.landed[at] = true;
        Ok(())
    }
}
