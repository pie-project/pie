//! Weights on the device: one host buffer per plan param, landed straight
//! from the checkpoint executor and uploaded by every program that reads
//! it. Dense planes of a plain dtype land as they are; an affine quantised
//! bank (`U4g64` and its 2/4/8-bit siblings: packed codes plus bf16
//! `.scales` and `.biases` params) is decoded on the host as it lands,
//! `w = scale · code + bias` rounded to bf16, so the kernels only ever see
//! dense bf16 planes.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::file::read::parse_metadata;
use checkpoint::file::serve;
use checkpoint::file::zt;
use checkpoint::plan::{StorageTarget, compile_streaming};
use checkpoint::serving::Stamp;
use checkpoint::types::BackendKind;
use kernels_cerebras::Tensor;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::Buffer;
use crate::device::Device;
use crate::error::{Fault, Result};
use crate::trace::{Handles, Root, Source, packed_row_bytes};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRow {
    Dense(Tensor),
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

/// Bytes a param occupies as the checkpoint lands it.
pub(crate) fn plane_bytes(name: &str, dtype: Dtype, shape: &[u64]) -> Result<u64> {
    let (rows, width) = rectangle(shape);
    if let Some(per_row) = packed_row_bytes(dtype, width) {
        return Ok(rows.saturating_mul(per_row));
    }
    let element = model_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Param {
        name: name.to_string(),
        why: "is declared in a packed storage element that has no element size",
    })?;
    Ok(rows.saturating_mul(width).saturating_mul(element))
}

/// An affine quantised code plane: `bits` per code, one bf16 scale and bias
/// per `group` codes along a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Quant {
    pub bits: u32,
    pub group: u32,
}

/// The affine bank form of a dtype, if it has one.
pub(crate) fn quant_of(dtype: Dtype) -> Option<Quant> {
    Some(match dtype {
        Dtype::U4g32 => Quant { bits: 4, group: 32 },
        Dtype::U4g64 => Quant { bits: 4, group: 64 },
        Dtype::U8g64 => Quant { bits: 8, group: 64 },
        Dtype::U2g32 => Quant { bits: 2, group: 32 },
        Dtype::U2g64 => Quant { bits: 2, group: 64 },
        Dtype::U2g128 => Quant {
            bits: 2,
            group: 128,
        },
        _ => return None,
    })
}

/// The dtype a param's buffer holds: bf16 for a decoded bank, else its own.
fn landed_dtype(dtype: Dtype) -> Dtype {
    if quant_of(dtype).is_some() {
        Dtype::Bf16
    } else {
        dtype
    }
}

/// Refuses a param whose dtype has neither a dense storage form on this
/// device nor a host decoding.
fn dense_only(name: &str, dtype: Dtype) -> Result<()> {
    if crate::device::element_of(dtype).is_none() && quant_of(dtype).is_none() {
        return Err(Fault::Param {
            name: name.to_string(),
            why: "is declared in a packed dtype this backend neither lands nor decodes \
                  (affine 2/4/8-bit banks decode to bf16; tiled, mxfp4 and K-quant do not)",
        });
    }
    Ok(())
}

fn round_bf16_bytes(v: f32) -> [u8; 2] {
    let u = v.to_bits();
    let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16;
    (r as u16).to_le_bytes()
}

fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])) << 16)
}

/// Decodes packed `codes` (`rows` rows of `width` codes, LSB-first) with
/// bf16 `scales` and `biases` (`[rows, width / group]`) into bf16 bytes.
pub(crate) fn dequantize(
    q: Quant,
    rows: usize,
    width: usize,
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
) -> Vec<u8> {
    let per_byte = (8 / q.bits) as usize;
    let mask = (1u32 << q.bits) - 1;
    let row_bytes = width.div_ceil(per_byte);
    let groups = width / q.group as usize;
    let mut out = Vec::with_capacity(rows * width * 2);
    for r in 0..rows {
        for c in 0..width {
            let byte = codes[r * row_bytes + c / per_byte];
            let code = (u32::from(byte) >> ((c % per_byte) as u32 * q.bits)) & mask;
            let g = r * groups + c / q.group as usize;
            let w = bf16_at(scales, g) * code as f32 + bf16_at(biases, g);
            out.extend_from_slice(&round_bf16_bytes(w));
        }
    }
    out
}

/// The scales and biases params of the bank `name`.
fn bank_members(name: &str) -> (String, String) {
    (
        format!("{name}{}", dtype::SCALES),
        format!("{name}{}", dtype::BIASES),
    )
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
        for param in &trace.params {
            dense_only(&param.name, param.dtype)?;
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
        let _ = &landing.attachments;

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
                 model",
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
                let elem = model_compiler::arena::elem_bytes(p.dtype).unwrap_or(0);
                (at, (adapters, slot, rows, cols, elem))
            })
            .collect();

        let streaming = compile_streaming(&metadata, contract, target)?;
        let mut sink = Landing {
            device,
            trace,
            index: &index,
            sizes: &sizes,
            buffers: (0..trace.params.len()).map(|_| None).collect(),
            held: BTreeMap::new(),
        };
        Execution::new(&streaming, snapshot)
            .streaming()
            .sink(&mut sink)
            .run()?;
        if let Some((&at, _)) = sink.held.iter().next() {
            return Err(Fault::Param {
                name: trace.params[at].name.clone(),
                why: "landed only part of its quantised bank; the load contract never \
                      published the rest",
            });
        }
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
            let (rows, width) = rectangle(&param.shape);
            table.push(Some(WeightRow::Dense(handles.root(Root {
                source: Source::Weight {
                    param: at as u32,
                    plane: 0,
                    transposed: false,
                },
                dtype: landed_dtype(param.dtype),
                rows: u32::try_from(rows).unwrap_or(u32::MAX),
                width: u32::try_from(width).unwrap_or(u32::MAX),
            }))));
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

struct Landing<'a> {
    device: &'a Device,
    trace: &'a Trace,
    index: &'a BTreeMap<&'a str, usize>,
    sizes: &'a [u64],
    buffers: Vec<Option<Buffer>>,
    /// Planes of quantised banks waiting for their partners, by param.
    held: BTreeMap<usize, Vec<u8>>,
}

impl Landing<'_> {
    /// The codes param of the bank `at` belongs to, if it is a bank member.
    fn bank_of(&self, at: usize) -> Option<usize> {
        let name = self.trace.params[at].name.as_str();
        if quant_of(self.trace.params[at].dtype).is_some() {
            return Some(at);
        }
        for suffix in [dtype::SCALES, dtype::BIASES] {
            if let Some(codes) = name.strip_suffix(suffix)
                && let Some(&c) = self.index.get(codes)
                && quant_of(self.trace.params[c].dtype).is_some()
            {
                return Some(c);
            }
        }
        None
    }

    /// Decodes the bank of `codes_at` once all three planes are held.
    fn decode(&mut self, codes_at: usize) -> std::result::Result<(), LoadError> {
        let param = &self.trace.params[codes_at];
        let (scales_name, biases_name) = bank_members(&param.name);
        let (Some(&s_at), Some(&b_at)) = (
            self.index.get(scales_name.as_str()),
            self.index.get(biases_name.as_str()),
        ) else {
            return Err(LoadError::Contract(format!(
                "`{}` is a quantised bank and the plan names no `{scales_name}` or `{biases_name}`",
                param.name
            )));
        };
        if !(self.held.contains_key(&codes_at)
            && self.held.contains_key(&s_at)
            && self.held.contains_key(&b_at))
        {
            return Ok(());
        }
        let codes = self.held.remove(&codes_at).expect("held");
        let scales = self.held.remove(&s_at).expect("held");
        let biases = self.held.remove(&b_at).expect("held");
        let q = quant_of(param.dtype).expect("a bank");
        let (rows, width) = rectangle(&param.shape);
        let (rows, width) = (rows as usize, width as usize);
        if width % q.group as usize != 0 {
            return Err(LoadError::Contract(format!(
                "`{}` has {width} codes per row, not a whole number of groups of {}",
                param.name, q.group
            )));
        }
        let groups = rows * (width / q.group as usize);
        if scales.len() != groups * 2 || biases.len() != groups * 2 {
            return Err(LoadError::Contract(format!(
                "`{}` has {groups} groups and its scales and biases land {} and {} bytes",
                param.name,
                scales.len(),
                biases.len()
            )));
        }
        let dense = dequantize(q, rows, width, &codes, &scales, &biases);
        let upload = |dtype: Dtype, rows: usize, width: usize, bytes: &[u8]| {
            self.device
                .upload(
                    dtype,
                    u32::try_from(rows).unwrap_or(u32::MAX),
                    u32::try_from(width).unwrap_or(u32::MAX),
                    bytes,
                )
                .map_err(|fault| LoadError::Internal(fault.to_string()))
        };
        self.buffers[codes_at] = Some(upload(Dtype::Bf16, rows, width, &dense)?);
        for (at, bytes) in [(s_at, &scales), (b_at, &biases)] {
            let p = &self.trace.params[at];
            let (r, w) = rectangle(&p.shape);
            self.buffers[at] = Some(upload(p.dtype, r as usize, w as usize, bytes)?);
        }
        Ok(())
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
        if let Some(codes_at) = self.bank_of(at) {
            self.held.insert(at, bytes.to_vec());
            return self.decode(codes_at);
        }
        let param = &self.trace.params[at];
        let (rows, width) = rectangle(&param.shape);
        self.buffers[at] = Some(
            self.device
                .upload(
                    param.dtype,
                    u32::try_from(rows).unwrap_or(u32::MAX),
                    u32::try_from(width).unwrap_or(u32::MAX),
                    bytes,
                )
                .map_err(|fault| LoadError::Internal(fault.to_string()))?,
        );
        Ok(())
    }
}

/// Weights for a dry shell (`crate::dry`): the handle table a landing would
/// mint, and no buffers.
impl Weights {
    pub(crate) fn dry(handles: &Handles, trace: &Trace, _prescale: bool) -> Result<Weights> {
        for param in &trace.params {
            dense_only(&param.name, param.dtype)?;
        }
        let sizes: Vec<u64> = trace
            .params
            .iter()
            .map(|p| plane_bytes(&p.name, p.dtype, &p.shape))
            .collect::<Result<_>>()?;
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
                        elem: model_compiler::arena::elem_bytes(param.dtype).unwrap_or(0),
                        host: Vec::new(),
                    },
                );
            }
            let (rows, width) = rectangle(&param.shape);
            table.push(Some(WeightRow::Dense(handles.root(Root {
                source: Source::Weight {
                    param: at as u32,
                    plane: 0,
                    transposed: false,
                },
                dtype: landed_dtype(param.dtype),
                rows: u32::try_from(rows).unwrap_or(u32::MAX),
                width: u32::try_from(width).unwrap_or(u32::MAX),
            }))));
        }
        Ok(Weights {
            buffers: (0..trace.params.len()).map(|_| None).collect(),
            table: WeightTable(table),
            banks,
            bytes: sizes.iter().sum(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_affine_bank_decodes_lsb_first_per_group() {
        // 2 rows x 4 codes (u4, group 64 does not divide 4; use U4g32 with width 32).
        let q = Quant { bits: 4, group: 32 };
        let (rows, width) = (1usize, 32usize);
        let codes: Vec<u8> = (0..16)
            .map(|i| ((2 * i + 1) << 4 | (2 * i)) as u8)
            .collect(); // codes 0..32 LSB-first
        let scales: Vec<u8> = round_bf16_bytes(0.5).to_vec();
        let biases: Vec<u8> = round_bf16_bytes(-1.0).to_vec();
        let out = dequantize(q, rows, width, &codes, &scales, &biases);
        let got: Vec<f32> = (0..width).map(|i| bf16_at(&out, i)).collect();
        let want: Vec<f32> = (0..width).map(|c| 0.5 * (c % 16) as f32 - 1.0).collect();
        assert_eq!(got, want);
    }
}
