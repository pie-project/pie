use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::{ModelContract, TensorContract};
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::file::Metadata;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve;
use checkpoint::file::serve::Artifact;
use checkpoint::file::zt;
use checkpoint::plan::{LoadPlan, StorageTarget, compile, compile_streaming};
use checkpoint::serving::{self, Stamp};
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use kernels_metal::Tensor;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::experts::{Attachments, Plan, Source, Tier};
use crate::gather;
use crate::host_source::HostSource;
use crate::mapping::{self, Mapping};
use crate::run::{WeightRow, WeightTable};
use crate::weight_store::Store;

pub(crate) const ALIGN: u64 = 256;

#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    pub bank: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Debug)]
pub struct Weights {
    store: Store,
    mapped: Vec<Buffer>,
    table: WeightTable,
    tier: Option<RefCell<Tier>>,
    rows: Option<RefCell<gather::Slab>>,
    banks: BTreeMap<String, Bank>,
    residue: (usize, u64),
    decoded: Vec<Buffer>,
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

fn banks(trace: &Trace, places: &[Place], seat: impl Fn(usize) -> u64) -> BTreeMap<String, Bank> {
    trace
        .params
        .iter()
        .zip(places)
        .enumerate()
        .filter(|(_, (param, _))| param.source == ParamSource::Registered)
        .map(|(at, (param, place))| {
            let adapters =
                u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(u32::MAX);
            let slot = if adapters == 0 {
                0
            } else {
                place.full / u64::from(adapters)
            };
            let (rows, cols) = rectangle(param.shape.get(1..).unwrap_or(&[]));
            (
                param.name.clone(),
                Bank {
                    offset: seat(at),
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

impl Weights {
    pub fn resident(
        device: &Context,
        handles: &Handles,
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        plan: &Plan,
    ) -> Result<Weights> {
        readable_plane_orders(trace)?;
        let gather = plan.gathered();

        serves_this_deployment(path, trace.platform.backend(), &trace.name)?;

        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };

        let target = StorageTarget::for_backend(BackendKind::Metal, 0, 1);
        let landing = compile(&metadata, contract, target.clone())?;

        let places = places(trace, plan, plan.gathered())?;
        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        let pairings = pairings(&landing, &index)?;

        match warm(
            device,
            handles,
            trace,
            contract,
            &metadata,
            snapshot,
            target.clone(),
            path,
            plan,
            &places,
            &pairings,
        ) {
            Ok(weights) => return Ok(weights),
            Err(None) => {}
            Err(Some(why)) => {
                let held = std::fs::metadata(path).map(|it| it.len()).unwrap_or(0);
                let staging = std::env::temp_dir();
                eprintln!(
                    "engine-metal: {} loads cold — {why}.\n\
                     engine-metal: that road reads all {:.2} GiB of it through the executor \
                     and stages {:.2} GiB of routed bands under {}, where {} free; the warm \
                     arm would have read no weight byte at all.",
                    path.display(),
                    held as f64 / (1u64 << 30) as f64,
                    plan.source_bytes() as f64 / (1u64 << 30) as f64,
                    staging.display(),
                    match crate::host_source::free_bytes(&staging) {
                        Some(free) => format!("{:.2} GiB is", free as f64 / (1u64 << 30) as f64),
                        None => "an unknown amount is".to_string(),
                    },
                );
            }
        }

        let spans: Vec<(u64, u64)> = places.iter().map(|p| (p.offset, p.reserved)).collect();
        let mut store = Store::zeroed(device, &spans, device.max_buffer())?;

        let mut host = HostSource::open(plan.source_bytes())?;
        let mut table = HostSource::open(gather.source_bytes())?;
        let mut sink = Landing {
            store: &mut store,
            host: &mut host,
            table: &mut table,
            plan,
            gather,
            places: &places,
            index: &index,
            landed: vec![false; places.len()],
        };
        let landed = if plan.streams() || gather.gathers() {
            let streaming = compile_streaming(&metadata, contract, target)?;
            Execution::new(&streaming, snapshot)
                .streaming()
                .sink(&mut sink)
                .run()?;
            sink.landed
        } else {
            let mut scratch = vec![0u8; usize::try_from(landing.memory.arena_bytes()).unwrap_or(0)];
            let mut backing: &mut [u8] = &mut scratch;
            Execution::new(&landing, snapshot)
                .arena(&mut backing)
                .sink(&mut sink)
                .run()?;
            let landed = sink.landed;
            drop(scratch);
            landed
        };

        let mut weight_table = Vec::with_capacity(places.len());
        for (at, place) in places.iter().enumerate() {
            if !landed[at] && trace.params[at].source == ParamSource::Checkpoint {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            let dense = |place: &Place| -> Result<Tensor> {
                Ok(Tensor::new(
                    store.bind(handles, place.offset, place.bytes)?,
                    place.rows,
                    place.width,
                    place.dtype,
                ))
            };
            weight_table.push(Some(match pairings.get(trace.params[at].name.as_str()) {
                Some(pairing) => WeightRow::Planes(kernels_metal::Bank {
                    codes: dense(place)?,
                    scales: dense(&places[pairing.scales])?,
                    biases: pairing.biases.map(|at| dense(&places[at])).transpose()?,
                    group: pairing.group,
                    bits: pairing.bits,
                }),
                None => WeightRow::Dense(dense(place)?),
            }));
        }
        let offsets: Vec<u64> = places.iter().map(|place| place.offset).collect();
        let tier = plan
            .streams()
            .then(|| {
                Tier::open(plan, &store, Source::landed(plan, host), &offsets).map(RefCell::new)
            })
            .transpose()?;
        let rows = gather
            .gathers()
            .then(|| {
                gather::Slab::open(
                    gather,
                    &store,
                    Source::from_host(table, gather.host_bands()),
                    &offsets,
                )
                .map(RefCell::new)
            })
            .transpose()?;
        Ok(Weights {
            store,
            mapped: Vec::new(),
            table: WeightTable(weight_table),
            tier,
            rows,
            banks: banks(trace, &places, |at| places[at].offset),
            residue: (0, 0),
            decoded: Vec::new(),
        })
    }

    #[must_use]
    pub fn warm(&self) -> bool {
        !self.mapped.is_empty()
    }

    #[must_use]
    pub fn windows(&self) -> usize {
        self.mapped.len()
    }

    #[must_use]
    pub fn residue(&self) -> (usize, u64) {
        self.residue
    }

    #[must_use]
    pub fn rows(&self) -> Option<&RefCell<gather::Slab>> {
        self.rows.as_ref()
    }

    #[must_use]
    pub fn tier(&self) -> Option<&RefCell<Tier>> {
        self.tier.as_ref()
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
    pub fn seats(&self) -> Vec<BankSeat> {
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
    pub fn table(&self) -> &WeightTable {
        &self.table
    }

    pub fn relabel_conv_weights(
        &mut self,
        device: &Context,
        handles: &Handles,
        trace: &Trace,
    ) -> Result<()> {
        for (at, param) in trace.params.iter().enumerate() {
            let model_ir::ParamLayout::ConvTapsMajor { c_in, taps } = param.layout else {
                continue;
            };
            let Some(Some(WeightRow::Dense(plane))) = self.table.0.get(at).copied() else {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a convolution weight that did not land as one dense plane",
                });
            };
            if plane.dtype != Dtype::Bf16 {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a convolution weight this plane relabels only in bf16",
                });
            }
            let pitch = (c_in as usize).saturating_mul(taps as usize);
            if pitch == 0 || plane.width as usize != pitch {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a convolution weight whose row is not `c_in * taps` wide",
                });
            }
            let rows = plane.rows as usize;
            let bytes = handles.read(plane.buf, (rows * pitch * 2) as u64)?;
            let mut relabelled = vec![0u8; bytes.len()];
            for n in 0..rows {
                let row = n * pitch;
                for tap in 0..taps as usize {
                    for c in 0..c_in as usize {
                        let from = (row + c * taps as usize + tap) * 2;
                        let into = (row + tap * c_in as usize + c) * 2;
                        relabelled[into..into + 2].copy_from_slice(&bytes[from..from + 2]);
                    }
                }
            }
            let mut buffer = Buffer::zeroed(device, relabelled.len() as u64)?;
            buffer.write(0, &relabelled)?;
            let handle = handles.bind(&buffer, 0, relabelled.len() as u64)?;
            self.table.0[at] = Some(WeightRow::Dense(Tensor::new(
                handle,
                plane.rows,
                plane.width,
                Dtype::Bf16,
            )));
            self.decoded.push(buffer);
        }
        Ok(())
    }

    pub fn decode_absorbed(
        &mut self,
        device: &Context,
        handles: &Handles,
        trace: &Trace,
    ) -> Result<()> {
        let started = std::time::Instant::now();
        let mut decoded_bytes = 0u64;
        let mut wanted = crate::decoded::absorbed_weights(trace);
        wanted.extend(crate::decoded::lane_axis_weights(trace));
        for at in wanted {
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
            decoded_bytes += plane.len() as u64;
            self.table.0[at] = Some(WeightRow::Dense(Tensor::new(
                handle,
                u32::try_from(n).unwrap_or(u32::MAX),
                u32::try_from(k).unwrap_or(u32::MAX),
                Dtype::Bf16,
            )));
            self.decoded.push(buffer);
        }
        if decoded_bytes > 0 && crate::diag::on().tier_trace {
            eprintln!(
                "load: decoded {:.2} GiB of absorbed banks to bf16 in {:.2} s",
                decoded_bytes as f64 / (1u64 << 30) as f64,
                started.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() + self.mapped.iter().map(Buffer::bytes).sum::<u64>()
    }
}

pub fn attachments(trace: &Trace, contract: &ModelContract, path: &Path) -> Result<Attachments> {
    readable_plane_orders(trace)?;
    let metadata = if path.is_dir() {
        parse_metadata(path)?
    } else {
        zt::parse(path)?
    };
    let landing = compile(
        &metadata,
        contract,
        StorageTarget::for_backend(BackendKind::Metal, 0, 1),
    )?;
    let index: BTreeMap<&str, usize> = trace
        .params
        .iter()
        .enumerate()
        .map(|(at, param)| (param.name.as_str(), at))
        .collect();
    let mut out = Attachments::new();
    for (name, pairing) in pairings(&landing, &index)? {
        let Some(&at) = index.get(name) else {
            continue;
        };
        let mut planes = vec![pairing.scales];
        planes.extend(pairing.biases);
        out.insert(at, planes);
    }
    Ok(out)
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
    offset: u64,
    bytes: u64,
    reserved: u64,
    full: u64,
    streamed: bool,
    gathered: bool,
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

#[derive(Debug, Clone, Copy)]
enum Seat {
    Artifact(u64),
    Store(u64),
}

const PROBE: usize = 32;

#[allow(clippy::too_many_arguments)]
fn warm(
    device: &Context,
    handles: &Handles,
    trace: &Trace,
    contract: &ModelContract,
    metadata: &Metadata,
    snapshot: &Path,
    target: StorageTarget,
    path: &Path,
    plan: &Plan,
    places: &[Place],
    pairings: &BTreeMap<&str, Pairing>,
) -> std::result::Result<Weights, Option<String>> {
    let gather = plan.gathered();
    if path.is_dir() {
        return Err(None);
    }
    let artifact = Artifact::open(path).map_err(|_| None)?;
    let spans = artifact.spans();
    let align = serving::alignment(&spans);
    if align == 0 || !align.is_multiple_of(ALIGN) {
        return Err(Some(format!(
            "its serving offsets are aligned to {align} and a matrix operand on this \
             device wants {ALIGN}"
        )));
    }
    let locate = |name: &str| -> Option<(u64, u64)> {
        let located = artifact.locate(name).ok()?;
        Some((located.at + located.plane.offset, located.plane.len))
    };

    let verbatim: BTreeMap<&str, &str> = contract
        .tensors
        .iter()
        .filter_map(|tensor| match &tensor.expr {
            checkpoint::contract::Expr::Src(from) => Some((tensor.name.as_str(), from.as_str())),
            _ => None,
        })
        .collect();

    let mut seats = Vec::with_capacity(places.len());
    let mut covers: Vec<(&str, u64, u64)> = Vec::with_capacity(places.len());
    let mut objects: Vec<&str> = Vec::with_capacity(places.len());
    let mut landed: Vec<TensorContract> = Vec::new();
    let mut store_bytes = 0u64;
    let mut residue_bytes = 0u64;
    let mut from_file: Vec<Option<u64>> = vec![None; places.len()];
    let mut bands: BTreeMap<usize, u64> = BTreeMap::new();
    let mut rows_of: BTreeMap<usize, u64> = BTreeMap::new();
    let mut stored = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let place = &places[index];
        if param.source == ParamSource::Registered {
            seats.push(Seat::Store(store_bytes));
            objects.push(param.name.as_str());
            store_bytes += place.reserved;
            continue;
        }
        let object = locate(&param.name)
            .map(|found| (param.name.as_str(), found))
            .or_else(|| {
                verbatim
                    .get(param.name.as_str())
                    .and_then(|from| locate(from).map(|found| (*from, found)))
            });
        let Some(object) = object else {
            if place.streamed {
                return Err(Some(format!(
                    "its {} `{}` is a plane this load holds CPU-side and the artifact \
                     carries in no form — its seat is a slab and cannot be landed whole \
                     into that seat",
                    if place.gathered {
                        "gathered table"
                    } else {
                        "routed band"
                    },
                    param.name,
                )));
            }
            let Some(residue_of) = residue_of(contract, &param.name) else {
                return Err(Some(format!(
                    "it carries no plane `{}` and this contract declares none — the \
                     artifact and the trace were not written from each other",
                    param.name,
                )));
            };
            landed.push(residue_of);
            seats.push(Seat::Store(store_bytes));
            objects.push(param.name.as_str());
            store_bytes += place.reserved;
            residue_bytes += place.full;
            continue;
        };
        let (object, (offset, length)) = object;
        if length != place.full || (!place.streamed && place.bytes != place.full) {
            return Err(Some(format!(
                "its plane `{}` (stored as `{object}`) is {length} bytes and this plan \
                 declares {}",
                param.name, place.full,
            )));
        }
        if !offset.is_multiple_of(ALIGN) {
            return Err(Some(format!(
                "its plane `{}` (stored as `{object}`) lies at {offset}, and a matrix \
                 operand on this device wants {ALIGN}-byte alignment",
                param.name,
            )));
        }
        from_file[index] = Some(offset);
        stored += place.full;
        objects.push(object);
        if place.streamed {
            if place.gathered {
                rows_of.insert(index, offset);
            } else {
                bands.insert(index, offset);
            }
            seats.push(Seat::Store(store_bytes));
            store_bytes += place.reserved;
            continue;
        }
        covers.push((param.name.as_str(), offset, place.full));
        seats.push(Seat::Artifact(offset));
    }

    if residue_bytes.saturating_mul(16) > stored {
        return Err(Some(format!(
            "{} of its {} plane(s) are ones this load computes rather than reads — {} \
             bytes against {stored} stored, which is not a warm load with a residue but \
             a cold load with a mapping",
            landed.len(),
            places.len(),
            residue_bytes,
        )));
    }
    let residue_planes = landed.len();

    let map = Mapping::of(path).map_err(|why| Some(why.to_string()))?;
    for (index, param) in trace.params.iter().enumerate() {
        let Some(offset) = from_file[index] else {
            continue;
        };
        let length = places[index].full;
        let from = usize::try_from(offset).unwrap_or(usize::MAX);
        let upto = usize::try_from(offset.saturating_add(length)).unwrap_or(usize::MAX);
        let mine = map.get(from..upto).ok_or_else(|| {
            Some(format!(
                "its plane `{}` lies at {offset}..{upto} and the file holds {} bytes",
                param.name,
                map.len(),
            ))
        })?;
        let published = artifact.plane(objects[index]).map_err(|why| {
            Some(format!(
                "its plane `{}` has no zero-copy view ({why})",
                param.name
            ))
        })?;
        let ends = |bytes: &[u8]| {
            let head = bytes
                .get(..PROBE.min(bytes.len()))
                .unwrap_or_default()
                .to_vec();
            let tail = bytes
                .get(bytes.len().saturating_sub(PROBE)..)
                .unwrap_or_default()
                .to_vec();
            (head, tail)
        };
        if published.len() != mine.len() || ends(published) != ends(mine) {
            return Err(Some(format!(
                "its manifest puts `{}` at {offset} and the bytes there are not the ones \
                 the container publishes for it — two readings of one offset that do not \
                 agree, so nothing here binds either",
                param.name,
            )));
        }
    }

    let spans: Vec<(u64, u64)> = places
        .iter()
        .enumerate()
        .filter_map(|(index, place)| match seats[index] {
            Seat::Store(offset) => Some((offset, place.reserved)),
            Seat::Artifact(_) => None,
        })
        .collect();
    debug_assert_eq!(
        spans
            .last()
            .map_or(0, |&(offset, reserved)| offset + reserved),
        store_bytes,
        "the store's spans pack to the bytes the arm counted"
    );
    let mut store =
        Store::zeroed(device, &spans, device.max_buffer()).map_err(|why| Some(why.to_string()))?;
    if !landed.is_empty() {
        let into: BTreeMap<&str, (u64, u64)> = trace
            .params
            .iter()
            .enumerate()
            .filter_map(|(index, param)| match seats[index] {
                Seat::Store(offset) if param.source != ParamSource::Registered => {
                    Some((param.name.as_str(), (offset, places[index].full)))
                }
                _ => None,
            })
            .collect();
        let only = ModelContract {
            alignment: contract.alignment,
            tensors: landed,
            groups: Vec::new(),
        };
        let plan = compile(metadata, &only, target).map_err(|why| {
            Some(format!(
                "the {} plane(s) it does not store do not compile ({why})",
                only.tensors.len(),
            ))
        })?;
        let mut scratch = vec![0u8; usize::try_from(plan.memory.arena_bytes()).unwrap_or(0)];
        let mut backing: &mut [u8] = &mut scratch;
        let mut sink = Residue {
            store: &mut store,
            into: &into,
            landed: 0,
        };
        Execution::new(&plan, snapshot)
            .arena(&mut backing)
            .sink(&mut sink)
            .run()
            .map_err(|why| {
                Some(format!(
                    "the plane(s) it does not store do not land ({why})"
                ))
            })?;
        if sink.landed != only.tensors.len() {
            return Err(Some(format!(
                "the residue landing published {} of {} plane(s)",
                sink.landed,
                only.tensors.len(),
            )));
        }
    }
    let cuts = mapping::cut(&map, mapping::ceiling(device.max_buffer()), &covers)
        .map_err(|why| Some(why.to_string()))?;
    let mut files = Vec::with_capacity(cuts.len());
    for cut in &cuts {
        files.push(
            Buffer::window(device, std::sync::Arc::clone(&map), *cut)
                .map_err(|why| Some(why.to_string()))?,
        );
    }
    let row = |index: usize| -> std::result::Result<Tensor, Option<String>> {
        let place = &places[index];
        let handle = match seats[index] {
            Seat::Artifact(offset) => {
                let at = cuts
                    .iter()
                    .rposition(|cut| cut.holds(offset, place.bytes))
                    .ok_or_else(|| {
                        Some(format!(
                            "`{}` lies at {offset} for {} bytes, which no one of this \
                             artifact's {} mapped window(s) holds whole",
                            trace.params[index].name,
                            place.bytes,
                            cuts.len(),
                        ))
                    })?;
                let view = cuts[at].view(offset).ok_or_else(|| {
                    Some(format!(
                        "`{}` does not seat in its window",
                        trace.params[index].name
                    ))
                })?;
                handles.bind(&files[at], view, place.bytes)
            }
            Seat::Store(offset) => store.bind(handles, offset, place.bytes),
        }
        .map_err(|why| {
            Some(format!(
                "`{}` does not bind ({why})",
                trace.params[index].name
            ))
        })?;
        Ok(Tensor::new(handle, place.rows, place.width, place.dtype))
    };
    let mut table = Vec::with_capacity(places.len());
    for (index, param) in trace.params.iter().enumerate() {
        table.push(Some(match pairings.get(param.name.as_str()) {
            Some(pairing) => WeightRow::Planes(kernels_metal::Bank {
                codes: row(index)?,
                scales: row(pairing.scales)?,
                biases: pairing.biases.map(row).transpose()?,
                group: pairing.group,
                bits: pairing.bits,
            }),
            None => WeightRow::Dense(row(index)?),
        }));
    }
    let seated: Vec<u64> = seats
        .iter()
        .map(|seat| match seat {
            Seat::Store(offset) | Seat::Artifact(offset) => *offset,
        })
        .collect();
    let tier = plan
        .streams()
        .then(|| {
            Tier::open(
                plan,
                &store,
                Source::artifact(std::sync::Arc::clone(&map), bands),
                &seated,
            )
            .map(RefCell::new)
        })
        .transpose()
        .map_err(|why| Some(format!("its routed tier does not open ({why})")))?;
    let rows = gather
        .gathers()
        .then(|| {
            gather::Slab::open(
                gather,
                &store,
                Source::artifact(std::sync::Arc::clone(&map), rows_of),
                &seated,
            )
            .map(RefCell::new)
        })
        .transpose()
        .map_err(|why| Some(format!("its gathered row slab does not open ({why})")))?;
    Ok(Weights {
        store,
        mapped: files,
        table: WeightTable(table),
        tier,
        rows,
        banks: banks(trace, places, |at| seated[at]),
        residue: (residue_planes, residue_bytes),
        decoded: Vec::new(),
    })
}

fn residue_of(contract: &ModelContract, name: &str) -> Option<TensorContract> {
    let entry = contract.tensors.iter().find(|it| it.name == name)?;
    if entry.scales.is_some() || entry.zero_points.is_some() {
        return None;
    }
    let mut names_another = false;
    entry.expr.visit(&mut |node| {
        if matches!(node, checkpoint::contract::Expr::Out(_)) {
            names_another = true;
        }
    });
    (!names_another).then(|| entry.clone())
}

struct Residue<'a> {
    store: &'a mut Store,
    into: &'a BTreeMap<&'a str, (u64, u64)>,
    landed: usize,
}

impl TensorSink for Residue<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoadError> {
        let Some((offset, full)) = self.into.get(name).copied() else {
            return Ok(());
        };
        if bytes.len() as u64 != full {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {full} — a plane read at \
                 the wrong width is a model that computes",
                bytes.len(),
            )));
        }
        self.store
            .write(offset, bytes)
            .map_err(|fault| LoadError::Internal(fault.to_string()))?;
        self.landed += 1;
        Ok(())
    }
}

fn places(trace: &Trace, plan: &Plan, gather: &gather::Plan) -> Result<Vec<Place>> {
    let bytes = plane_bytes(trace)?;
    let mut out = Vec::with_capacity(trace.params.len());
    let mut at = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let (rows, width) = rectangle(&param.shape);
        let full = bytes[index];
        let seated = plan.resident(index);
        let gathered = gather.resident(index);
        let (reserve, rows) = match seated.or(gathered) {
            Some(slots) if rows > 0 => (full / rows * u64::from(slots), u64::from(slots)),
            _ => (full, rows),
        };
        out.push(Place {
            offset: at,
            bytes: reserve,
            reserved: reserve.next_multiple_of(ALIGN),
            full,
            streamed: seated.is_some() || gathered.is_some(),
            gathered: gathered.is_some(),
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype: param.dtype,
        });
        at += reserve.next_multiple_of(ALIGN);
    }
    Ok(out)
}

fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

struct Landing<'a> {
    store: &'a mut Store,
    host: &'a mut [u8],
    table: &'a mut [u8],
    plan: &'a Plan,
    gather: &'a gather::Plan,
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
        if bytes.len() as u64 != place.full {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {} — a plane read \
                 at the wrong width is a model that computes",
                bytes.len(),
                place.full
            )));
        }
        if place.streamed {
            let (what, at_host, into) = if place.gathered {
                (
                    "gathered table",
                    self.gather.host_at(at).unwrap_or(0),
                    &mut *self.table,
                )
            } else {
                (
                    "streamed band",
                    self.plan.host_at(at).unwrap_or(0),
                    &mut *self.host,
                )
            };
            let from = usize::try_from(at_host).unwrap_or(usize::MAX);
            let into = into.get_mut(from..from + bytes.len()).ok_or_else(|| {
                LoadError::Internal(format!(
                    "`{name}` is a {what} whose {} bytes leave its host source at \
                         offset {from}",
                    bytes.len()
                ))
            })?;
            into.copy_from_slice(bytes);
        } else {
            self.store
                .write(place.offset, bytes)
                .map_err(|fault| LoadError::Internal(fault.to_string()))?;
        }
        self.landed[at] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use checkpoint::contract::Expr;
    use checkpoint::types::{DType, Encoding};
    use model_ir::Platform;

    use super::*;

    #[test]
    fn weights_every_case() {
        the_residue_is_a_leaf_and_a_chain_is_not();
        the_store_is_laid_out_aligned_disjoint_and_in_plan_order();
    }

    fn the_residue_is_a_leaf_and_a_chain_is_not() {
        let entry = |name: &str, expr: Expr| TensorContract {
            name: name.to_string(),
            expr,
            shape: None,
            encoding: Encoding::Raw(DType::F32),
            scales: None,
            zero_points: None,
            visibility: Default::default(),
        };
        let contract = ModelContract {
            alignment: 256,
            tensors: vec![
                entry(
                    "leaf",
                    Expr::Cast {
                        src: Box::new(Expr::Src("stored".into())),
                        to: Encoding::Raw(DType::F32),
                    },
                ),
                entry(
                    "chained",
                    Expr::Cast {
                        src: Box::new(Expr::Out("leaf".into())),
                        to: Encoding::Raw(DType::F32),
                    },
                ),
            ],
            groups: Vec::new(),
        };

        assert!(
            residue_of(&contract, "leaf").is_some(),
            "a cast of one stored tensor is a plane this arm can land by itself"
        );
        assert!(
            residue_of(&contract, "chained").is_none(),
            "an entry that names another entry is a chain, and the cold path is what \
             runs chains"
        );
        assert!(
            residue_of(&contract, "nowhere").is_none(),
            "a plane the contract does not declare is not a residue at all"
        );
    }

    fn the_store_is_laid_out_aligned_disjoint_and_in_plan_order() {
        let trace = models::sku("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships the SKU")
            .trace;
        let trace = trace(Platform::Metal);
        let places = places(&trace, &Plan::default(), &gather::Plan::default())
            .expect("every param of a bf16 SKU has an element size");

        assert_eq!(places.len(), trace.params.len());
        let mut end = 0u64;
        for (place, param) in places.iter().zip(&trace.params) {
            assert!(
                place.offset >= end,
                "`{}` overlaps its predecessor",
                param.name
            );
            assert_eq!(place.offset % ALIGN, 0, "`{}` is misaligned", param.name);
            assert!(place.bytes > 0, "`{}` reserves nothing", param.name);
            end = place.offset + place.reserved;
        }

        assert_eq!(places[0].offset, 0);
        assert_eq!(places[0].rows, 248_320);
        assert_eq!(places[0].width, 1024);
        assert_eq!(places[0].bytes, 248_320 * 1024 * 2);
    }
}
