//! An adapter bundle: one safetensors file whose tensors are named by the
//! registered banks they fill (`layers.{l}.{stanza}.{site}.{bank}`). A guest
//! attaches it to a pass with `kernel::adapter(bundle)`, its bytes the seed of
//! one channel, and an engine writes it into one adapter slot, keyed by its
//! digest so that every pass attaching the same bytes shares that slot. The
//! reading and the refusals are the same on every engine; each engine zeroes
//! the banks a bundle does not bring when it writes the slot. A bundle holds
//! each bank whole, and a rank of a tensor-parallel deployment takes the share
//! of it its bank holds.

use std::borrow::Cow;

use eta_compiler::codegen::launch::{LaunchOp, LaunchPackage, ValueOrigin};
use eta_ir::op::tags;
use poem_ir::{Dtype, Shard};

/// The sink name `kernel::adapter(bundle)` emits.
pub const SINK: &str = "adapter";

/// A registered bank as an engine seats it, one slot of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bank<'a> {
    /// The bank's name: its struct path in the model text.
    pub name: &'a str,
    /// The bank's first axis after the slot axis.
    pub rows: u64,
    /// The product of the bank's axes after `rows`.
    pub cols: u64,
    /// The bytes one slot spans.
    pub slot: u64,
    /// The element type the bank holds.
    pub dtype: Dtype,
    /// How the ranks cut the bank, as the trace this rank runs states it:
    /// its axis counts the slot axis, and its segments are this rank's.
    pub shard: &'a Shard,
}

/// Which rank of how many an engine is in the ranks its deployment is split
/// across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank {
    pub rank: u32,
    pub world: u32,
}

impl Rank {
    /// The one rank of a deployment no ranks split.
    pub const WHOLE: Rank = Rank { rank: 0, world: 1 };
}

/// Why an engine will not bind a bundle: the bank or tensor it is about
/// ([`SINK`] when it is the sink or the file itself), and a clause that
/// follows that name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// The bank, tensor or sink the refusal names.
    pub bank: String,
    /// What is wrong with it.
    pub why: String,
}

/// A bundle read off the bytes its channel carried.
pub struct Bundle<'a> {
    /// The file: the channel's bytes up to the end of the tensor data, so the
    /// channel's zero padding is no part of it.
    file: &'a [u8],
    tensors: Vec<Tensor<'a>>,
}

/// One tensor as the bundle's header states it.
struct Tensor<'a> {
    name: String,
    dtype: Element,
    shape: Vec<u64>,
    bytes: &'a [u8],
}

/// The element types a bundle may carry.
#[derive(Clone, Copy)]
enum Element {
    F32,
    Bf16,
    F16,
}

impl Element {
    fn parse(spelled: &str) -> Option<Element> {
        match spelled {
            "F32" => Some(Element::F32),
            "BF16" => Some(Element::Bf16),
            "F16" => Some(Element::F16),
            _ => None,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Element::F32 => 4,
            Element::Bf16 | Element::F16 => 2,
        }
    }

    fn value(self, bytes: &[u8]) -> f32 {
        match self {
            Element::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Element::Bf16 => {
                f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
            }
            Element::F16 => f16(u16::from_le_bytes([bytes[0], bytes[1]])),
        }
    }

    fn is(self, dtype: Dtype) -> bool {
        matches!(
            (self, dtype),
            (Element::F32, Dtype::F32) | (Element::Bf16, Dtype::Bf16) | (Element::F16, Dtype::F16)
        )
    }
}

impl std::fmt::Debug for Bundle<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bundle")
            .field("bytes", &self.file.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

fn refused(bank: &str, why: impl Into<String>) -> Refusal {
    Refusal {
        bank: bank.to_string(),
        why: why.into(),
    }
}

/// The channel whose seed is the bundle a program's `adapter` sink attaches,
/// if the program calls it. A pass carries one adapter, so a second call is
/// refused.
pub fn channel_of(package: &LaunchPackage) -> Result<Option<u32>, Refusal> {
    let mut bundles =
        package
            .stages
            .iter()
            .flat_map(|stage| &stage.ops)
            .filter(|op: &&LaunchOp| {
                op.tag == tags::SINK_CALL
                    && package
                        .names
                        .get(op.name_index as usize)
                        .map(String::as_str)
                        == Some(SINK)
            });
    let Some(call) = bundles.next() else {
        return Ok(None);
    };
    if bundles.next().is_some() {
        return Err(refused(
            SINK,
            "is called twice, and a pass carries one adapter",
        ));
    }
    let [value] = call.args.as_slice() else {
        return Err(refused(
            SINK,
            format!(
                "takes {} arguments, and `kernel::adapter(bundle)` takes one",
                call.args.len()
            ),
        ));
    };
    package
        .values
        .iter()
        .find(|declared| declared.id == *value)
        .filter(|declared| {
            matches!(
                declared.source,
                ValueOrigin::ChannelRead | ValueOrigin::ChannelTake
            )
        })
        .map(|declared| Some(declared.channel))
        .ok_or_else(|| {
            refused(
                SINK,
                "reads a value that is not a channel, and a bundle is a channel's seed",
            )
        })
}

impl<'a> Bundle<'a> {
    /// Reads the safetensors header of `wire`, the bytes a channel carried.
    /// Refuses a file whose header does not parse, a tensor whose data lies
    /// outside the file or disagrees with its shape, and any element type
    /// but F32, BF16 and F16, naming the tensor.
    pub fn read(wire: &'a [u8]) -> Result<Bundle<'a>, Refusal> {
        let unreadable =
            |why: &str| refused(SINK, format!("the bundle is not a safetensors file: {why}"));
        let head = wire
            .first_chunk::<8>()
            .ok_or_else(|| unreadable("it is shorter than its 8-byte header length"))?;
        let start = usize::try_from(u64::from_le_bytes(*head))
            .ok()
            .and_then(|len| len.checked_add(8))
            .filter(|start| *start <= wire.len())
            .ok_or_else(|| unreadable("its header runs past the bytes sent"))?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&wire[8..start])
                .map_err(|_| unreadable("its header is not a JSON object"))?;
        let data = &wire[start..];
        let mut end = start;
        let mut tensors = Vec::with_capacity(header.len());
        for (name, info) in header {
            if name == "__metadata__" {
                continue;
            }
            let spelled = info["dtype"].as_str().unwrap_or_default();
            let dtype = Element::parse(spelled).ok_or_else(|| {
                refused(
                    &name,
                    format!("is {spelled:?} in the bundle, and a bundle carries F32, BF16 or F16"),
                )
            })?;
            let shape: Option<Vec<u64>> = info["shape"]
                .as_array()
                .and_then(|dims| dims.iter().map(serde_json::Value::as_u64).collect());
            let span: Option<Vec<usize>> = info["data_offsets"].as_array().and_then(|span| {
                span.iter()
                    .map(|at| at.as_u64().and_then(|at| usize::try_from(at).ok()))
                    .collect()
            });
            let (Some(shape), Some(&[from, to])) = (shape, span.as_deref()) else {
                return Err(refused(
                    &name,
                    "has no readable `shape` or `data_offsets` in the bundle header",
                ));
            };
            let bytes = data.get(from..to).ok_or_else(|| {
                refused(
                    &name,
                    format!("has data {from}..{to} outside the bundle's data"),
                )
            })?;
            let elems = shape
                .iter()
                .try_fold(1u64, |all, dim| all.checked_mul(*dim));
            if elems.and_then(|elems| elems.checked_mul(dtype.bytes() as u64))
                != Some(bytes.len() as u64)
            {
                return Err(refused(
                    &name,
                    format!(
                        "has shape {shape:?} of {spelled} and {} bytes of data",
                        bytes.len()
                    ),
                ));
            }
            end = end.max(start + to);
            tensors.push(Tensor {
                name,
                dtype,
                shape,
                bytes,
            });
        }
        Ok(Bundle {
            file: &wire[..end],
            tensors,
        })
    }

    /// The slot key: the blake3 digest of the file.
    #[must_use]
    pub fn key(&self) -> [u8; 32] {
        *blake3::hash(self.file).as_bytes()
    }

    /// One whole-slot plane per tensor, in the dtype of the bank it names.
    /// Each tensor is the bank whole, so on `rank` it is first cut to the
    /// rank's share of the bank's cut. Refuses, naming the tensor, one that
    /// names no bank of `banks` (the bundle was trained for another program)
    /// or whose shape is not that bank's whole without the slot axis.
    pub fn planes(
        &self,
        banks: &[Bank<'_>],
        rank: Rank,
    ) -> Result<Vec<(String, Vec<u8>)>, Refusal> {
        self.tensors
            .iter()
            .map(|tensor| {
                let refuse = |why: String| refused(&tensor.name, why);
                let bank = banks
                    .iter()
                    .find(|bank| bank.name == tensor.name)
                    .ok_or_else(|| {
                        refuse(
                            "is not a registered bank of this deployment, so the bundle \
                             was trained for another program"
                                .to_string(),
                        )
                    })?;
                let (shape, bytes) = tensor.share(bank, rank).map_err(refuse)?;
                let (rows, cols) = match shape.split_first() {
                    Some((rows, rest)) => (*rows, rest.iter().product()),
                    None => (1, 1),
                };
                if (rows, cols) != (bank.rows, bank.cols) {
                    let share = if shape == tensor.shape {
                        String::new()
                    } else {
                        format!(", {shape:?} on this rank,")
                    };
                    return Err(refuse(format!(
                        "has shape {:?}{share} and the bank holds {} x {} values a slot",
                        tensor.shape, bank.rows, bank.cols
                    )));
                }
                let mut plane = vec![0u8; usize::try_from(bank.slot).unwrap_or(usize::MAX)];
                let values = bytes.chunks_exact(tensor.dtype.bytes());
                let wrote = match bank.dtype {
                    dtype if tensor.dtype.is(dtype) => plane
                        .get_mut(..bytes.len())
                        .map(|to| to.copy_from_slice(&bytes)),
                    Dtype::Bf16 => plane.get_mut(..values.len() * 2).map(|to| {
                        for (to, value) in to.as_chunks_mut::<2>().0.iter_mut().zip(values) {
                            to.copy_from_slice(&bf16(tensor.dtype.value(value)).to_le_bytes());
                        }
                    }),
                    Dtype::F32 => plane.get_mut(..values.len() * 4).map(|to| {
                        for (to, value) in to.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                            to.copy_from_slice(&tensor.dtype.value(value).to_le_bytes());
                        }
                    }),
                    dtype => {
                        return Err(refuse(format!(
                            "seats {dtype:?} elements, and a bundle converts into bf16 or \
                             f32 banks"
                        )));
                    }
                };
                wrote.ok_or_else(|| {
                    refuse(format!(
                        "carries more values than the bank's {} bytes a slot",
                        bank.slot
                    ))
                })?;
                Ok((tensor.name.clone(), plane))
            })
            .collect()
    }
}

impl<'a> Tensor<'a> {
    /// This tensor, which holds `bank` whole, cut to the share `rank` holds:
    /// along the bank's cut, each segment's band of the whole, a segment of
    /// fewer heads than ranks held whole by each rank of its group, as
    /// `poem::import` bands a checkpoint. A bank no ranks cut is the whole.
    fn share(&self, bank: &Bank<'_>, rank: Rank) -> Result<(Vec<u64>, Cow<'a, [u8]>), String> {
        let Shard::Cut {
            axis,
            segments,
            heads,
        } = bank.shard
        else {
            return Ok((self.shape.clone(), Cow::Borrowed(self.bytes)));
        };
        if rank.world <= 1 {
            return Ok((self.shape.clone(), Cow::Borrowed(self.bytes)));
        }
        // A bank's first axis is its slot axis, which a tensor has not.
        let at = axis
            .checked_sub(1)
            .map(|at| at as usize)
            .filter(|at| *at < self.shape.len())
            .ok_or_else(|| {
                format!(
                    "has shape {:?} and the bank is cut on its axis {axis}, which is no \
                     axis of the tensor after the slot axis",
                    self.shape
                )
            })?;
        if rank.rank >= rank.world {
            return Err(format!(
                "is cut across {} ranks, and this engine is rank {}",
                rank.world, rank.rank
            ));
        }
        let world = u64::from(rank.world);
        let mut bands = Vec::with_capacity(segments.len());
        let mut extent = 0u64;
        for (i, segment) in segments.iter().enumerate() {
            let parts = Shard::parts(heads.get(i).copied(), world)
                .map_err(|why| format!("is cut across ranks, and {why}"))?;
            let group = u64::from(rank.rank) / (world / parts);
            bands.push((extent + group * segment, *segment));
            extent += segment * parts;
        }
        if self.shape[at] != extent {
            return Err(format!(
                "has shape {:?} and the bank holds {extent} values whole along axis {at}, \
                 which {world} ranks cut",
                self.shape
            ));
        }
        let mut shape = self.shape.clone();
        shape[at] = segments.iter().sum();
        if self.bytes.is_empty() {
            return Ok((shape, Cow::Borrowed(self.bytes)));
        }
        // No axis is zero, so no product of axes holds more than the file.
        let outer: u64 = self.shape[..at].iter().product();
        let inner = self.shape[at + 1..].iter().product::<u64>() as usize * self.dtype.bytes();
        let row = inner * extent as usize;
        let mut bytes = Vec::with_capacity(self.bytes.len() / rank.world as usize);
        for o in 0..outer as usize {
            for (start, len) in &bands {
                let from = o * row + *start as usize * inner;
                bytes.extend_from_slice(&self.bytes[from..from + *len as usize * inner]);
            }
        }
        Ok((shape, Cow::Owned(bytes)))
    }
}

/// `value` rounded to the nearest bf16, ties to even.
fn bf16(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// The f32 an IEEE half-precision `bits` stands for.
fn f16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}
