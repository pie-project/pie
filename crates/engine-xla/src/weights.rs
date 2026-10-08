//! Weights on the device: one PJRT buffer per plan param, landed straight
//! from the checkpoint executor. Everything is resident — there is no host
//! tier to stream from, so a model that does not fit is refused at load.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve;
use checkpoint::file::zt;
use checkpoint::plan::{LoadPlan, StorageTarget, compile_streaming};
use checkpoint::serving::Stamp;
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use kernels_xla::{Bank, Tensor};
use poem_ir::{Dtype, Fused, ParamSource, Trace};

use crate::device::Device;
use crate::error::{Fault, Result};
use crate::pjrt::Buffer;
use crate::trace::{Handles, Root, Source, packed_row_bytes};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRow {
    Dense(Tensor),

    Planes(Bank),
}

#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    pub bank: &'a str,

    pub bytes: &'a [u8],
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

#[derive(Debug, Clone)]
struct AdapterBank {
    param: usize,
    adapters: u32,
    slot: u64,
    rows: u64,
    cols: u64,
    elem: u64,
    /// The whole bank on the host; a registration rewrites one slot of it
    /// and lands the bank again.
    host: Vec<u8>,
}

pub struct Weights {
    buffers: Vec<Option<Buffer>>,
    table: WeightTable,
    banks: BTreeMap<String, AdapterBank>,
    bytes: u64,
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights")
            .field("params", &self.buffers.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// The shape a param lands as: its leading axis by the product of the rest.
fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

/// The rectangle a quantized bank's plane lands as: one row per weight row
/// (the contraction axis minor), so a routed bank `[E, N, K]` is `[E·N, K]`
/// and its kernels read it without a relayout (an mxfp4 codes plane declares
/// its row as `[blocks, 16]` bytes, merged back into one row).
fn bank_rectangle(dtype: Dtype, shape: &[u64]) -> (u64, u64) {
    let tail = if dtype == Dtype::Mxfp4 { 2 } else { 1 };
    if shape.len() <= tail {
        return rectangle(shape);
    }
    let (lead, row) = shape.split_at(shape.len() - tail);
    (lead.iter().product(), row.iter().product())
}

/// Bytes a param occupies as the checkpoint lands it.
pub(crate) fn plane_bytes(name: &str, dtype: Dtype, shape: &[u64]) -> Result<u64> {
    let (rows, width) = rectangle(shape);
    if let Some(per_row) = packed_row_bytes(dtype, width) {
        return Ok(rows.saturating_mul(per_row));
    }
    let element = poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Param {
        name: name.to_string(),
        why: "is declared in a packed storage element that has no element size",
    })?;
    Ok(rows.saturating_mul(width).saturating_mul(element))
}

impl Weights {
    pub fn resident(
        device: &Device,
        handles: &Handles,
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        device_cap: u64,
    ) -> Result<Weights> {
        if let Some(param) = trace.params.iter().find(|p| p.dtype == Dtype::U4g64tiled) {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is declared U4g64tiled, the CUDA fragment order; convert the \
                      artifact for this backend",
            });
        }
        serves_this_deployment(path, trace.platform.backend(), &trace.name)?;

        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };
        let target = StorageTarget::for_backend(BackendKind::Xla, 0, 1);
        let landing = checkpoint::plan::compile(&metadata, contract, target.clone())?;
        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        let pairings = pairings(&landing, &index)?;
        tracing::debug!(
            pairings = pairings.len(),
            attachments = landing.attachments.len(),
            "xla weights: quantized banks"
        );
        if crate::serve::timing() {
            eprintln!(
                "xla weights: {} attachments, {} pairings",
                landing.attachments.len(),
                pairings.len()
            );
        }

        let sizes: Vec<u64> = trace
            .params
            .iter()
            .map(|p| plane_bytes(&p.name, p.dtype, &p.shape))
            .collect::<Result<_>>()?;
        let total: u64 = sizes.iter().sum();
        if total > device_cap {
            return Err(Fault::Residency(format!(
                "the weights are {} MiB and the device budget for them is {} MiB; this \
                 backend keeps every plane resident — shrink the pools or serve a smaller \
                 quantization",
                total >> 20,
                device_cap >> 20
            )));
        }

        let banks_of: BTreeMap<usize, (u32, u64, u64, u64, u64)> = trace
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.source == ParamSource::Registered)
            .map(|(at, p)| {
                let adapters = u32::try_from(p.shape.first().copied().unwrap_or(0)).unwrap_or(0);
                let slot = if adapters == 0 {
                    0
                } else {
                    sizes[at] / u64::from(adapters)
                };
                let (rows, cols) = rectangle(p.shape.get(1..).unwrap_or(&[]));
                let elem = poem_compiler::arena::elem_bytes(p.dtype).unwrap_or(0);
                (at, (adapters, slot, rows, cols, elem))
            })
            .collect();

        // An mxfp4 bank lands pre-scaled (e5m2 weights, `kernels_xla::pack`)
        // when the weights still fit at its doubled size; otherwise, or when
        // a bank's scales leave e5m2's exact range, as native e2m1 codes.
        let mut prescale: BTreeMap<usize, usize> = BTreeMap::new();
        for (name, pairing) in &pairings {
            let at = index[name];
            if trace.params[at].dtype == Dtype::Mxfp4 && pairing.biases.is_none() {
                prescale.insert(at, pairing.scales);
            }
        }
        let extra: u64 = prescale.keys().map(|&at| sizes[at]).sum();
        if std::env::var_os("PIE_XLA_MXFP4_CODES").is_some() || total + extra > device_cap {
            prescale.clear();
        }
        let scales_of: BTreeMap<usize, usize> = prescale.iter().map(|(&c, &s)| (s, c)).collect();
        let mut bank_planes = std::collections::BTreeSet::new();
        for (name, pairing) in &pairings {
            bank_planes.insert(index[name]);
            bank_planes.insert(pairing.scales);
            bank_planes.extend(pairing.biases);
        }
        let shape_of = |at: usize| -> (u64, u64) {
            let p = &trace.params[at];
            if bank_planes.contains(&at) {
                bank_rectangle(p.dtype, &p.shape)
            } else {
                rectangle(&p.shape)
            }
        };

        let transposed = gemm_only(trace, &bank_planes);
        let codes_planes: std::collections::BTreeSet<usize> =
            pairings.keys().map(|name| index[name]).collect();
        // A gather-only table lands folded, `[V/128, 128·row]` (codes as
        // their packed bytes, scales and zero points alike): see
        // `gather_only`.
        let mut folded: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
        for at in gather_only(trace, &codes_planes) {
            let pairing = pairings[trace.params[at].name.as_str()];
            for plane in [Some(at), Some(pairing.scales), pairing.biases]
                .into_iter()
                .flatten()
            {
                let p = &trace.params[plane];
                let (rows, width) = bank_rectangle(p.dtype, &p.shape);
                let row = packed_row_bytes(p.dtype, width).unwrap_or(width);
                folded.insert(plane, fold_of(rows, row));
            }
        }

        let streaming = compile_streaming(&metadata, contract, target)?;
        let mut sink = Landing {
            device,
            trace,
            index: &index,
            sizes: &sizes,
            buffers: (0..trace.params.len()).map(|_| None).collect(),
            stored: trace.params.iter().map(|p| p.dtype).collect(),
            prescale: &prescale,
            scales_of: &scales_of,
            held: BTreeMap::new(),
            shape_of: &shape_of,
            transposed: &transposed,
            folded: &folded,
        };
        Execution::new(&streaming, snapshot)
            .streaming()
            .sink(&mut sink)
            .run()?;
        if let Some(&at) = sink.held.keys().next() {
            return Err(Fault::Param {
                name: trace.params[at].name.clone(),
                why: "is half of an mxfp4 bank whose other plane never landed",
            });
        }
        let stored = sink.stored;
        // A pre-scaled mxfp4 plane holds a byte per code, twice its codes.
        let total = total
            + (0..stored.len())
                .filter(|&at| stored[at] == Dtype::E5m2 && trace.params[at].dtype != Dtype::E5m2)
                .map(|at| sizes[at])
                .sum::<u64>();
        let mut buffers = sink.buffers;

        let mut banks = BTreeMap::new();
        for (&at, &(adapters, slot, rows, cols, elem)) in &banks_of {
            let param = &trace.params[at];
            let (r, w) = rectangle(&param.shape);
            let host = vec![0u8; usize::try_from(sizes[at]).unwrap_or(0)];
            buffers[at] = Some(device.upload(
                param.dtype,
                u32::try_from(r).unwrap_or(u32::MAX),
                u32::try_from(w).unwrap_or(u32::MAX),
                &host,
            )?);
            banks.insert(
                param.name.clone(),
                AdapterBank {
                    param: at,
                    adapters,
                    slot,
                    rows,
                    cols,
                    elem,
                    host,
                },
            );
        }

        let mut table = Vec::with_capacity(trace.params.len());
        for (at, param) in trace.params.iter().enumerate() {
            if buffers[at].is_none() {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            let dense = |at: usize| -> Tensor {
                let p = &trace.params[at];
                if let Some(&(rows, width)) = folded.get(&at) {
                    return handles.root(Root {
                        source: Source::Weight {
                            param: at as u32,
                            plane: 0,
                            transposed: false,
                        },
                        dtype: stored[at],
                        rows: u32::try_from(rows).unwrap_or(u32::MAX),
                        width: u32::try_from(width).unwrap_or(u32::MAX),
                    });
                }
                let (rows, width) = shape_of(at);
                // A pre-scaled mxfp4 bank is one e5m2 weight per code.
                let width = if stored[at] == p.dtype {
                    width
                } else {
                    kernels_xla::pack::codes_per_row(p.dtype, width).unwrap_or(width)
                };
                handles.root(Root {
                    source: Source::Weight {
                        param: at as u32,
                        plane: 0,
                        transposed: transposed.contains(&at),
                    },
                    dtype: stored[at],
                    rows: u32::try_from(rows).unwrap_or(u32::MAX),
                    width: u32::try_from(width).unwrap_or(u32::MAX),
                })
            };
            table.push(Some(match pairings.get(param.name.as_str()) {
                Some(pairing) => WeightRow::Planes(Bank {
                    codes: dense(at),
                    scales: dense(pairing.scales),
                    biases: pairing.biases.map(dense),
                    group: pairing.group,
                    bits: pairing.bits,
                }),
                None => WeightRow::Dense(dense(at)),
            }));
        }
        Ok(Weights {
            buffers,
            table: WeightTable(table),
            banks,
            bytes: total,
        })
    }

    #[must_use]
    pub fn table(&self) -> &WeightTable {
        &self.table
    }

    #[must_use]
    pub fn buffer(&self, param: u32) -> Option<&Buffer> {
        self.buffers.get(param as usize)?.as_ref()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn register_adapter(
        &mut self,
        device: &Device,
        trace: &Trace,
        id: u32,
        planes: &[AdapterPlane<'_>],
    ) -> Result<()> {
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
                        "seats {} adapters while this registration is id {id}",
                        bank.adapters
                    ),
                });
            }
            if plane.bytes.len() as u64 != bank.slot {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} bytes per adapter while this plane carries {}",
                        bank.slot,
                        plane.bytes.len()
                    ),
                });
            }
        }
        for plane in planes {
            let bank = self.banks.get_mut(plane.bank).expect("checked above");
            let at = (u64::from(id) * bank.slot) as usize;
            bank.host[at..at + plane.bytes.len()].copy_from_slice(plane.bytes);
            let param = &trace.params[bank.param];
            let (r, w) = rectangle(&param.shape);
            self.buffers[bank.param] = Some(device.upload(
                param.dtype,
                u32::try_from(r).unwrap_or(u32::MAX),
                u32::try_from(w).unwrap_or(u32::MAX),
                &bank.host,
            )?);
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
                              them to be factors of",
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
                          and this plan names no zero points for the other half",
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

/// The dense bf16 params every reader of which is a gemm taking it as its
/// weight (`y = act · wᵀ`). They land transposed, `[K, N]` with the output
/// axis minor: XLA's TPU gemm streams a `[N, K]` weight at ~1.2-1.4 TB/s
/// once a fire has more than one row, and a `[K, N]` one near the HBM
/// roofline (a single-row gemv is at the roofline either way). So do the
/// sub-byte codes planes of quantized gemm weights the TPU would otherwise
/// lay out K-minor (below). A param read any other way (an embedding
/// gather, a norm, a routed bank) lands as declared.
/// `PIE_XLA_WEIGHT_T=0` lands every param as declared.
fn gemm_only(
    trace: &Trace,
    banks: &std::collections::BTreeSet<usize>,
) -> std::collections::BTreeSet<usize> {
    use poem_ir::{Def, Linear, Operands, Operation};
    let mut out = std::collections::BTreeSet::new();
    if std::env::var("PIE_XLA_WEIGHT_T").is_ok_and(|v| v == "0") {
        return out;
    }
    let param_of = |v: poem_ir::ValueId| match trace.values.get(v.0 as usize).map(|d| &d.def) {
        Some(Def::Weight(p)) => Some(*p as usize),
        _ => None,
    };
    let mut other = std::collections::BTreeSet::new();
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        inputs.clear();
        node.op.inputs(&mut inputs);
        let w = match &node.op {
            Operation::Linear(Linear::Matmul { w, .. } | Linear::LmHead { w, .. })
            | Operation::Fused(
                Fused::MatmulBias { w, .. }
                | Fused::MatmulGeglu { w, .. }
                | Fused::LmHeadSoftcap { w, .. },
            ) => Some(*w),
            _ => None,
        };
        for &v in &inputs {
            let Some(p) = param_of(v) else { continue };
            if Some(v) == w {
                out.insert(p);
            } else {
                other.insert(p);
            }
        }
    }
    for decl in &trace.values {
        if let Def::Merge(arms) = &decl.def {
            other.extend(arms.iter().filter_map(|(v, _)| param_of(*v)));
        }
    }
    out.retain(|&p| {
        let param = &trace.params[p];
        if other.contains(&p) || param.source == ParamSource::Registered {
            return false;
        }
        if banks.contains(&p) {
            // A bank's codes plane `[N, K]` whose K tiles evenly and whose N
            // does not: the TPU would lay it out K-minor, and the grouped
            // dot relayouts it every fire (gpt-oss's o_proj `[2880, 4096]`:
            // 45 us a layer). It lands `[K, N]` with N padded to whole
            // 128-lane tiles (`crate::trace` slices the padding off).
            let (n, k) = bank_rectangle(param.dtype, &param.shape);
            let k = kernels_xla::pack::codes_per_row(param.dtype, k).unwrap_or(k);
            return kernels_xla::pack::code_bits(param.dtype).is_some_and(|b| b < 8)
                && param.dtype != Dtype::Mxfp4
                && k.is_multiple_of(CODES_TILE)
                && !n.is_multiple_of(CODES_TILE);
        }
        param.dtype == Dtype::Bf16 && param.shape.len() == 2
    });
    out
}

/// Rows a transposed sub-byte codes plane pads its output axis to.
pub(crate) const CODES_TILE: u64 = 128;

/// Rows a gather-only table pads to, a multiple of every fold.
const FOLD: u64 = 128;

/// The `[rows / f, f · row]` a gather-only plane of `rows` rows of `row`
/// elements lands as (rows padded to whole `FOLD`s): the least `f` whose
/// stored row tiles evenly (128 lanes) while the stored rows do not, so the
/// TPU lays it out row-major and a gather reads a short stored row per id
/// (gpt-oss's embedding: codes 4 rows of 1440 bytes a stored row, scales
/// 128 rows of 45).
fn fold_of(rows: u64, row: u64) -> (u64, u64) {
    let rows = rows.div_ceil(FOLD) * FOLD;
    let folds = || (0..=7).map(|s| 1u64 << s);
    let f = folds()
        .find(|f| (f * row).is_multiple_of(128) && !(rows / f).is_multiple_of(128))
        .or_else(|| folds().find(|f| (f * row).is_multiple_of(128)))
        .unwrap_or(FOLD);
    (rows / f, f * row)
}

/// The sub-byte bank codes planes every reader of which is an embedding
/// gather. They land folded, codes as the packed bytes of their rows
/// (`u8 [V/128, 128·bytes]`), scales and zero points `[V/128, 128·groups]`
/// (`kernels_xla::layout::folding`): the TPU lays a 2-D array out with the
/// axis that tiles evenly minor, so a `[V, row]` table lands column-major and
/// every gather relayouts all of it first (gpt-oss's 201088-row embedding:
/// 1.75 ms a step as `u4` codes, 0.78 ms as bytes). `PIE_XLA_EMBED_CODES=1`
/// lands them as codes.
fn gather_only(
    trace: &Trace,
    codes: &std::collections::BTreeSet<usize>,
) -> std::collections::BTreeSet<usize> {
    use poem_ir::{Def, Layout, Operands, Operation};
    let mut out = std::collections::BTreeSet::new();
    if std::env::var_os("PIE_XLA_EMBED_CODES").is_some_and(|v| v != "0") {
        return out;
    }
    let param_of = |v: poem_ir::ValueId| match trace.values.get(v.0 as usize).map(|d| &d.def) {
        Some(Def::Weight(p)) => Some(*p as usize),
        _ => None,
    };
    let mut other = std::collections::BTreeSet::new();
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        inputs.clear();
        node.op.inputs(&mut inputs);
        let table = match &node.op {
            Operation::Layout(Layout::Embed { table, .. } | Layout::EmbedConcat { table, .. }) => {
                Some(*table)
            }
            _ => None,
        };
        for &v in &inputs {
            let Some(p) = param_of(v) else { continue };
            if Some(v) == table {
                out.insert(p);
            } else {
                other.insert(p);
            }
        }
    }
    for decl in &trace.values {
        if let Def::Merge(arms) = &decl.def {
            other.extend(arms.iter().filter_map(|(v, _)| param_of(*v)));
        }
    }
    out.retain(|&p| {
        let param = &trace.params[p];
        !other.contains(&p)
            && codes.contains(&p)
            && param.source != ParamSource::Registered
            && matches!(kernels_xla::pack::code_bits(param.dtype), Some(2 | 4))
            && param.dtype != Dtype::Mxfp4
    });
    out
}

/// A row-major `[rows, cols]` array of 2-byte elements as `[cols, rows]`,
/// in cache-sized blocks over a few threads.
fn transpose_2byte(bytes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    const B: usize = 64;
    let mut dst = vec![0u8; bytes.len()];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(16);
    // Each thread fills whole output rows (input columns) `c0..c1`.
    let per = cols.div_ceil(threads).div_ceil(B) * B;
    std::thread::scope(|scope| {
        for (t, out) in dst.chunks_mut((2 * per * rows).max(2)).enumerate() {
            scope.spawn(move || {
                let c0 = t * per;
                let c1 = (c0 + per).min(cols);
                for rb in (0..rows).step_by(B) {
                    for cb in (c0..c1).step_by(B) {
                        for c in cb..(cb + B).min(c1) {
                            let o = &mut out[2 * (c - c0) * rows..2 * (c - c0 + 1) * rows];
                            for r in rb..(rb + B).min(rows) {
                                let at = 2 * (r * cols + c);
                                o[2 * r..2 * r + 2].copy_from_slice(&bytes[at..at + 2]);
                            }
                        }
                    }
                }
            });
        }
    });
    dst
}

struct Landing<'a> {
    device: &'a Device,
    trace: &'a Trace,
    index: &'a BTreeMap<&'a str, usize>,
    sizes: &'a [u64],
    buffers: Vec<Option<Buffer>>,
    /// The dtype each param landed as (its declared one, or `E5m2` for a
    /// pre-scaled mxfp4 bank's codes).
    stored: Vec<Dtype>,
    /// mxfp4 codes param -> its scales param, for the banks landing
    /// pre-scaled; and the reverse.
    prescale: &'a BTreeMap<usize, usize>,
    scales_of: &'a BTreeMap<usize, usize>,
    /// The first-published plane of a pre-scaled bank, until its partner
    /// lands.
    held: BTreeMap<usize, Vec<u8>>,
    /// The rectangle each param lands as.
    shape_of: &'a dyn Fn(usize) -> (u64, u64),
    /// The params that land transposed (`gemm_only`).
    transposed: &'a std::collections::BTreeSet<usize>,
    /// The planes of gather-only tables: the `[rows, width]` each lands
    /// folded as (`gather_only`).
    folded: &'a BTreeMap<usize, (u64, u64)>,
}

impl Landing<'_> {
    fn upload(&mut self, at: usize, dtype: Dtype, width: u64, bytes: &[u8]) -> Result<()> {
        let (rows, _) = (self.shape_of)(at);
        self.buffers[at] = Some(self.device.upload(
            dtype,
            u32::try_from(rows).unwrap_or(u32::MAX),
            u32::try_from(width).unwrap_or(u32::MAX),
            bytes,
        )?);
        self.stored[at] = dtype;
        Ok(())
    }

    /// Lands a param as declared; a bank's codes one code per byte; a
    /// gemm-only weight as `[width, rows]`.
    fn plain(&mut self, at: usize, bytes: &[u8]) -> Result<()> {
        let dtype = self.trace.params[at].dtype;
        let (rows, width) = (self.shape_of)(at);
        if self.transposed.contains(&at)
            && let Some(codes) = kernels_xla::pack::land(dtype, bytes)
        {
            // `[N, K]` codes, one per byte, as `[K, N']`, N padded to whole
            // tiles with zero codes.
            let k = codes.len() / rows as usize;
            let n = rows as usize;
            let padded = n.div_ceil(CODES_TILE as usize) * CODES_TILE as usize;
            let mut t = vec![0u8; k * padded];
            for (r, row) in codes.chunks_exact(k).enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    t[c * padded + r] = v;
                }
            }
            self.buffers[at] = Some(self.device.upload(
                dtype,
                u32::try_from(k).unwrap_or(u32::MAX),
                u32::try_from(padded).unwrap_or(u32::MAX),
                &t,
            )?);
            return Ok(());
        }
        if self.transposed.contains(&at) {
            let t = transpose_2byte(bytes, rows as usize, width as usize);
            self.buffers[at] = Some(self.device.upload(
                dtype,
                u32::try_from(width).unwrap_or(u32::MAX),
                u32::try_from(rows).unwrap_or(u32::MAX),
                &t,
            )?);
            return Ok(());
        }
        if let Some(&(rows, fold_width)) = self.folded.get(&at) {
            // Row-major `[V, row]` bytes are `[V/F, F·row]` as they stand;
            // only the padding to whole folds is new.
            let dtype = if kernels_xla::pack::code_elem(dtype).is_some() {
                Dtype::U8
            } else {
                dtype
            };
            let mut padded = bytes.to_vec();
            let elem = poem_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u64;
            padded.resize((rows * fold_width * elem) as usize, 0);
            self.buffers[at] = Some(self.device.upload(
                dtype,
                u32::try_from(rows).unwrap_or(u32::MAX),
                u32::try_from(fold_width).unwrap_or(u32::MAX),
                &padded,
            )?);
            self.stored[at] = dtype;
            return Ok(());
        }
        match kernels_xla::pack::land(dtype, bytes) {
            Some(codes) => self.upload(at, dtype, width, &codes),
            None => self.upload(at, dtype, width, bytes),
        }
    }

    /// Lands an mxfp4 bank's codes as e5m2 weights, or as codes when a
    /// scale leaves e5m2's exact range.
    fn prescaled(&mut self, codes_at: usize, codes: &[u8], scales: &[u8]) -> Result<()> {
        let dtype = self.trace.params[codes_at].dtype;
        let (_, width) = (self.shape_of)(codes_at);
        match kernels_xla::pack::mxfp4_e5m2(codes, scales) {
            Some(w) => {
                let per_row = kernels_xla::pack::codes_per_row(dtype, width).unwrap_or(width);
                self.upload(codes_at, Dtype::E5m2, per_row, &w)
            }
            None => self.plain(codes_at, codes),
        }
    }
}

impl TensorSink for Landing<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoadError> {
        let at = *self.index.get(name).ok_or_else(|| {
            LoadError::Contract(format!(
                "the load contract publishes `{name}`, which this plan does not name"
            ))
        })?;
        if bytes.len() as u64 != self.sizes[at] {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {}",
                bytes.len(),
                self.sizes[at]
            )));
        }
        let landed = if let Some(&scales_at) = self.prescale.get(&at) {
            match self.held.remove(&scales_at) {
                Some(scales) => self.prescaled(at, bytes, &scales),
                None => {
                    self.held.insert(at, bytes.to_vec());
                    Ok(())
                }
            }
        } else if let Some(&codes_at) = self.scales_of.get(&at) {
            self.plain(at, bytes)
                .and_then(|()| match self.held.remove(&codes_at) {
                    Some(codes) => self.prescaled(codes_at, &codes, bytes),
                    None => {
                        self.held.insert(at, bytes.to_vec());
                        Ok(())
                    }
                })
        } else {
            self.plain(at, bytes)
        };
        landed.map_err(|fault| LoadError::Internal(fault.to_string()))
    }
}

/// Weights for a dry shell (`crate::dry`): the handle table a landing would
/// mint, and no buffers. A quantized param `X` pairs with the params
/// `X.scales` (and `X.biases`) the plan declares beside it, as the landing
/// plan's attachments pair them; an mxfp4 bank lands pre-scaled when
/// `prescale` (the choice a device with room makes), else as codes.
impl Weights {
    pub(crate) fn dry(handles: &Handles, trace: &Trace, prescale: bool) -> Result<Weights> {
        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        let sizes: Vec<u64> = trace
            .params
            .iter()
            .map(|p| plane_bytes(&p.name, p.dtype, &p.shape))
            .collect::<Result<_>>()?;
        let mut pairings: BTreeMap<usize, Pairing> = BTreeMap::new();
        for (at, param) in trace.params.iter().enumerate() {
            let Some(&scales) = index.get(poem_scales(&param.name).as_str()) else {
                continue;
            };
            if trace.params[scales].dtype == param.dtype && param.dtype == Dtype::Bf16 {
                continue;
            }
            let (group, bits) = match param.dtype {
                Dtype::Mxfp4 => (32, 4),
                Dtype::Nvfp4 => (16, 4),
                Dtype::U4g32 => (32, 4),
                Dtype::U4g64 | Dtype::U4g64tiled => (64, 4),
                Dtype::U8g64 => (64, 8),
                Dtype::U2g32 => (32, 2),
                Dtype::U2g64 => (64, 2),
                Dtype::U2g128 => (128, 2),
                _ => (0, 0),
            };
            pairings.insert(
                at,
                Pairing {
                    scales,
                    biases: index.get(poem_biases(&param.name).as_str()).copied(),
                    group,
                    bits,
                },
            );
        }
        let mut bank_planes = std::collections::BTreeSet::new();
        for (&at, pairing) in &pairings {
            bank_planes.insert(at);
            bank_planes.insert(pairing.scales);
            bank_planes.extend(pairing.biases);
        }
        let shape_of = |at: usize| -> (u64, u64) {
            let p = &trace.params[at];
            if bank_planes.contains(&at) {
                bank_rectangle(p.dtype, &p.shape)
            } else {
                rectangle(&p.shape)
            }
        };
        let transposed = gemm_only(trace, &bank_planes);
        let codes_planes: std::collections::BTreeSet<usize> = pairings.keys().copied().collect();
        let mut folded: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
        for at in gather_only(trace, &codes_planes) {
            let pairing = pairings[&at];
            for plane in [Some(at), Some(pairing.scales), pairing.biases]
                .into_iter()
                .flatten()
            {
                let p = &trace.params[plane];
                let (rows, width) = bank_rectangle(p.dtype, &p.shape);
                let row = packed_row_bytes(p.dtype, width).unwrap_or(width);
                folded.insert(plane, fold_of(rows, row));
            }
        }
        let mut stored: Vec<Dtype> = trace.params.iter().map(|p| p.dtype).collect();
        for (&at, pairing) in &pairings {
            if prescale && trace.params[at].dtype == Dtype::Mxfp4 && pairing.biases.is_none() {
                stored[at] = Dtype::E5m2;
            }
        }
        for &at in folded.keys() {
            if kernels_xla::pack::code_elem(trace.params[at].dtype).is_some() {
                stored[at] = Dtype::U8;
            }
        }
        let mut banks = BTreeMap::new();
        let mut table = Vec::with_capacity(trace.params.len());
        for (at, param) in trace.params.iter().enumerate() {
            if param.source == ParamSource::Registered {
                let adapters =
                    u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(0);
                let (rows, cols) = rectangle(param.shape.get(1..).unwrap_or(&[]));
                banks.insert(
                    param.name.clone(),
                    AdapterBank {
                        param: at,
                        adapters,
                        slot: if adapters == 0 {
                            0
                        } else {
                            sizes[at] / u64::from(adapters)
                        },
                        rows,
                        cols,
                        elem: poem_compiler::arena::elem_bytes(param.dtype).unwrap_or(0),
                        host: Vec::new(),
                    },
                );
            }
            let dense = |at: usize| -> Tensor {
                let p = &trace.params[at];
                let (rows, width) = match folded.get(&at) {
                    Some(&folded) => folded,
                    None => {
                        let (rows, width) = shape_of(at);
                        let width = if stored[at] == p.dtype || folded.contains_key(&at) {
                            width
                        } else {
                            kernels_xla::pack::codes_per_row(p.dtype, width).unwrap_or(width)
                        };
                        (rows, width)
                    }
                };
                handles.root(Root {
                    source: Source::Weight {
                        param: at as u32,
                        plane: 0,
                        transposed: transposed.contains(&at),
                    },
                    dtype: stored[at],
                    rows: u32::try_from(rows).unwrap_or(u32::MAX),
                    width: u32::try_from(width).unwrap_or(u32::MAX),
                })
            };
            table.push(Some(match pairings.get(&at) {
                Some(pairing) => WeightRow::Planes(Bank {
                    codes: dense(at),
                    scales: dense(pairing.scales),
                    biases: pairing.biases.map(dense),
                    group: pairing.group,
                    bits: pairing.bits,
                }),
                None => WeightRow::Dense(dense(at)),
            }));
        }
        Ok(Weights {
            buffers: (0..trace.params.len()).map(|_| None).collect(),
            table: WeightTable(table),
            banks,
            bytes: sizes.iter().sum(),
        })
    }
}

fn poem_scales(of: &str) -> String {
    format!("{of}{}", dtype::SCALES)
}

fn poem_biases(of: &str) -> String {
    format!("{of}{}", dtype::BIASES)
}
